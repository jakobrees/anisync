import datetime as dt
import random
import string
from collections import Counter

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app.config import get_settings
from app.db import get_db
from app.models import (
    Room,
    RoomMember,
    RoomPreferenceSubmission,
    RoomVote,
    User,
)
from app.realtime import manager
from app.security import get_current_user, hash_password, verify_password
from app.services.recommender import compute_recommendations, compute_vote_summary


settings = get_settings()
app = FastAPI(title="AniSync API")


app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    https_only=settings.cookie_secure,
    same_site=settings.cookie_same_site,
    max_age=60 * 60 * 24 * 14,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In local development, FastAPI serves preprocessed local image files.
# In production, catalog image paths may be Supabase public URLs instead.
if settings.resolved_media_root.exists():
    app.mount("/media", StaticFiles(directory=settings.resolved_media_root), name="media")


ALLOWED_ROOM_TYPES = {"TV", "MOVIE", "OVA", "ONA", "SPECIAL"}


class RegisterIn(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RoomCreateIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class JoinRoomIn(BaseModel):
    code: str = Field(min_length=1, max_length=20)


class SubmissionIn(BaseModel):
    query_text: str = Field(min_length=1, max_length=2000)


class ConstraintsIn(BaseModel):
    hard_constraint_year_start: int
    hard_constraint_year_end: int
    hard_constraint_allowed_types: list[str]
    reset_to_defaults: bool = False


class VoteIn(BaseModel):
    catalog_item_ids: list[int]


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def current_year() -> int:
    return utcnow().year


def bump_room(room: Room) -> None:
    """
    Every shared room-state mutation must increment state_revision.
    """
    room.state_revision += 1
    room.last_activity_at = utcnow()


def require_user(request: Request, db: Session) -> User:
    return get_current_user(request, db)


def get_room_by_code(db: Session, code: str) -> Room:
    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    return room


def is_room_member(db: Session, room_id: int, user_id: int) -> bool:
    return db.scalar(
        select(RoomMember.id).where(RoomMember.room_id == room_id, RoomMember.user_id == user_id)
    ) is not None


def require_room_member(db: Session, room: Room, user: User) -> None:
    if not is_room_member(db, room.id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this room.")


def require_host(room: Room, user: User) -> None:
    if room.host_user_id != user.id:
        raise HTTPException(status_code=403, detail="Only the room host can do this.")


def generate_room_code(db: Session) -> str:
    alphabet = string.ascii_uppercase + string.digits

    for _ in range(100):
        code = "".join(random.choice(alphabet) for _ in range(6))
        exists = db.scalar(select(Room.id).where(Room.code == code))
        if not exists:
            return code

    raise RuntimeError("Could not generate unique room code.")


def public_results(results_json: dict | None) -> dict | None:
    """
    Remove private compute data before sending results to room clients.

    We do not expose:
    - private raw query texts
    - query embeddings
    - per-user top-100 retrieval lists
    - private candidate lists
    """
    if not results_json:
        return None

    return {
        "chosen_k": results_json.get("chosen_k"),
        "kmeans_silhouette": results_json.get("kmeans_silhouette"),
        "eligible_catalog_subset_size": results_json.get("eligible_catalog_subset_size"),
        "applied_host_constraints": results_json.get("applied_host_constraints"),
        "clusters": results_json.get("clusters", []),
        "final_recommendations": results_json.get("final_recommendations", []),
        "vote_result_summary": results_json.get("vote_result_summary", []),
    }


def serialize_room(db: Session, room: Room, user: User) -> dict:
    users = list(
        db.execute(
            select(User, RoomMember)
            .join(RoomMember, RoomMember.user_id == User.id)
            .where(RoomMember.room_id == room.id)
            .order_by(RoomMember.joined_at)
        )
    )

    submissions = list(
        db.scalars(select(RoomPreferenceSubmission).where(RoomPreferenceSubmission.room_id == room.id))
    )
    submitted_user_ids = {submission.user_id for submission in submissions}

    own_submission = next((submission for submission in submissions if submission.user_id == user.id), None)

    voted_user_ids = set(
        db.scalars(
            select(distinct(RoomVote.user_id)).where(RoomVote.room_id == room.id)
        )
    )

    own_vote_ids = list(
        db.scalars(
            select(RoomVote.catalog_item_id).where(
                RoomVote.room_id == room.id,
                RoomVote.user_id == user.id,
            )
        )
    )

    participants = [
        {
            "user_id": member_user.id,
            "display_name": member_user.display_name,
            "is_host": member_user.id == room.host_user_id,
            "has_submitted": member_user.id in submitted_user_ids,
            "has_voted": member_user.id in voted_user_ids,
        }
        for member_user, _member in users
    ]

    return {
        "id": room.id,
        "code": room.code,
        "title": room.title,
        "category": room.category,
        "status": room.status,
        "state_revision": room.state_revision,
        "last_activity_at": room.last_activity_at.isoformat(),
        "is_host": room.host_user_id == user.id,
        "host_user_id": room.host_user_id,
        "constraints": {
            "hard_constraint_year_start": room.hard_constraint_year_start,
            "hard_constraint_year_end": room.hard_constraint_year_end,
            "hard_constraint_allowed_types": room.hard_constraint_allowed_types_json,
        },
        "participants": participants,
        "own_submission": own_submission.query_text if own_submission else "",
        "own_vote_catalog_item_ids": own_vote_ids,
        "vote_progress": {
            "voted_count": len(voted_user_ids),
            "member_count": len(participants),
            "pending_count": max(len(participants) - len(voted_user_ids), 0),
        },
        "results": public_results(room.results_json),
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "anisync-api"}


@app.post("/api/auth/register")
def register(payload: RegisterIn, request: Request, db: Session = Depends(get_db)) -> dict:
    existing = db.scalar(select(User).where(User.email == payload.email.lower()))
    if existing:
        raise HTTPException(status_code=400, detail="That email is already registered.")

    user = User(
        email=payload.email.lower(),
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    request.session["user_id"] = user.id

    return {"user": {"id": user.id, "email": user.email, "display_name": user.display_name}}


@app.post("/api/auth/login")
def login(payload: LoginIn, request: Request, db: Session = Depends(get_db)) -> dict:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    request.session["user_id"] = user.id

    return {"user": {"id": user.id, "email": user.email, "display_name": user.display_name}}


@app.post("/api/auth/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)
    return {"user": {"id": user.id, "email": user.email, "display_name": user.display_name}}


@app.get("/api/rooms")
def list_rooms(request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)

    rows = list(
        db.execute(
            select(Room)
            .join(RoomMember, RoomMember.room_id == Room.id)
            .where(RoomMember.user_id == user.id)
            .order_by(Room.last_activity_at.desc())
        )
    )

    rooms = [
        {
            "code": room.code,
            "title": room.title,
            "status": room.status,
            "state_revision": room.state_revision,
            "last_activity_at": room.last_activity_at.isoformat(),
        }
        for (room,) in rows
    ]

    return {"rooms": rooms}


@app.post("/api/rooms")
async def create_room(payload: RoomCreateIn, request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)

    year = current_year()
    room = Room(
        code=generate_room_code(db),
        title=payload.title.strip(),
        host_user_id=user.id,
        status="open",
        hard_constraint_year_start=year - 10,
        hard_constraint_year_end=year,
        hard_constraint_allowed_types_json=["TV"],
    )
    db.add(room)
    db.flush()

    db.add(RoomMember(room_id=room.id, user_id=user.id))
    bump_room(room)

    db.commit()
    db.refresh(room)

    return {"room": serialize_room(db, room, user)}


@app.post("/api/rooms/join")
async def join_room(payload: JoinRoomIn, request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, payload.code)

    joined_now = False
    if not is_room_member(db, room.id, user.id):
        db.add(RoomMember(room_id=room.id, user_id=user.id))
        bump_room(room)
        joined_now = True
        db.commit()
        db.refresh(room)

    if joined_now:
        await manager.broadcast(
            room.code,
            event_type="member_joined",
            state_revision=room.state_revision,
            changed_sections=["participants", "header"],
        )

    return {"room": serialize_room(db, room, user)}


@app.get("/api/rooms/{code}")
async def get_room(code: str, request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, code)

    # Join-by-URL behavior: a logged-in user who opens a valid room URL joins automatically.
    joined_now = False
    if not is_room_member(db, room.id, user.id):
        db.add(RoomMember(room_id=room.id, user_id=user.id))
        bump_room(room)
        joined_now = True
        db.commit()
        db.refresh(room)

    if joined_now:
        await manager.broadcast(
            room.code,
            event_type="member_joined",
            state_revision=room.state_revision,
            changed_sections=["participants", "header"],
        )

    return {"room": serialize_room(db, room, user)}


@app.post("/api/rooms/{code}/submit")
async def submit_preference(
    code: str,
    payload: SubmissionIn,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, code)
    require_room_member(db, room, user)

    query_text = payload.query_text.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Please enter your anime preference before submitting.")

    submission = db.scalar(
        select(RoomPreferenceSubmission).where(
            RoomPreferenceSubmission.room_id == room.id,
            RoomPreferenceSubmission.user_id == user.id,
        )
    )

    if submission:
        submission.query_text = query_text
        submission.updated_at = utcnow()
    else:
        db.add(RoomPreferenceSubmission(room_id=room.id, user_id=user.id, query_text=query_text))

    submitted_count = db.scalar(
        select(func.count(RoomPreferenceSubmission.id)).where(RoomPreferenceSubmission.room_id == room.id)
    )

    if submitted_count and submitted_count >= 1 and room.status == "open":
        room.status = "preferences_submitted"

    bump_room(room)
    db.commit()
    db.refresh(room)

    await manager.broadcast(
        room.code,
        event_type="preference_saved",
        state_revision=room.state_revision,
        changed_sections=["participants", "preference", "header"],
    )

    return {"room": serialize_room(db, room, user)}


@app.post("/api/rooms/{code}/constraints")
async def update_constraints(
    code: str,
    payload: ConstraintsIn,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, code)
    require_room_member(db, room, user)
    require_host(room, user)

    if room.status in {"results_ready", "voting_open", "voting_complete"}:
        raise HTTPException(status_code=400, detail="Constraints cannot be changed after computation starts.")

    year = current_year()

    if payload.reset_to_defaults:
        start_year = year - 10
        end_year = year
        allowed_types = ["TV"]
    else:
        start_year = payload.hard_constraint_year_start
        end_year = payload.hard_constraint_year_end
        allowed_types = [item.upper() for item in payload.hard_constraint_allowed_types]

    if start_year > end_year or start_year < 1960 or end_year > year + 1:
        raise HTTPException(status_code=400, detail="Please enter a valid inclusive year range.")

    if not allowed_types:
        raise HTTPException(status_code=400, detail="Select at least one allowed distribution type.")

    if any(item not in ALLOWED_ROOM_TYPES for item in allowed_types):
        raise HTTPException(status_code=400, detail="One or more selected types are invalid.")

    room.hard_constraint_year_start = start_year
    room.hard_constraint_year_end = end_year
    room.hard_constraint_allowed_types_json = sorted(set(allowed_types))
    bump_room(room)

    db.commit()
    db.refresh(room)

    await manager.broadcast(
        room.code,
        event_type="constraints_updated",
        state_revision=room.state_revision,
        changed_sections=["constraints", "header"],
    )

    return {"room": serialize_room(db, room, user)}


@app.post("/api/rooms/{code}/compute")
async def compute_room(code: str, request: Request, db: Session = Depends(get_db)) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, code)
    require_room_member(db, room, user)
    require_host(room, user)

    # Broadcast compute-start state first.
    bump_room(room)
    db.commit()
    db.refresh(room)

    await manager.broadcast(
        room.code,
        event_type="compute_started",
        state_revision=room.state_revision,
        changed_sections=["header"],
    )

    try:
        results = compute_recommendations(db, room)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    room.results_json = results
    room.status = "voting_open"
    bump_room(room)

    db.commit()
    db.refresh(room)

    await manager.broadcast(
        room.code,
        event_type="results_ready",
        state_revision=room.state_revision,
        changed_sections=["header", "results", "participants"],
    )

    return {"room": serialize_room(db, room, user)}


@app.post("/api/rooms/{code}/vote")
async def vote(
    code: str,
    payload: VoteIn,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    user = require_user(request, db)
    room = get_room_by_code(db, code)
    require_room_member(db, room, user)

    if not room.results_json:
        raise HTTPException(status_code=400, detail="Recommendations are not ready yet.")

    selected_ids = sorted(set(payload.catalog_item_ids))
    if not selected_ids:
        raise HTTPException(status_code=400, detail="Select at least one anime before submitting votes.")

    final_ids = {
        int(item["catalog_item_id"])
        for item in room.results_json.get("final_recommendations", [])
    }

    if any(item_id not in final_ids for item_id in selected_ids):
        raise HTTPException(status_code=400, detail="You can only vote on anime in the final recommendation list.")

    db.execute(delete(RoomVote).where(RoomVote.room_id == room.id, RoomVote.user_id == user.id))

    for item_id in selected_ids:
        db.add(RoomVote(room_id=room.id, user_id=user.id, catalog_item_id=item_id))

    # Important:
    # SessionLocal uses autoflush=False, so pending delete/insert vote changes
    # are not automatically visible to the count queries below.
    # Flush explicitly before checking whether everyone has voted.
    db.flush()

    member_count = db.scalar(select(func.count(RoomMember.id)).where(RoomMember.room_id == room.id)) or 0
    voted_user_count = db.scalar(
        select(func.count(distinct(RoomVote.user_id))).where(RoomVote.room_id == room.id)
    ) or 0

    changed_sections = ["participants", "results"]

    if member_count > 0 and voted_user_count >= member_count:
        vote_rows = list(
            db.execute(
                select(RoomVote.catalog_item_id, func.count(RoomVote.user_id))
                .where(RoomVote.room_id == room.id)
                .group_by(RoomVote.catalog_item_id)
            )
        )
        votes_by_item_id = Counter({int(item_id): int(count) for item_id, count in vote_rows})

        updated_results = dict(room.results_json)
        updated_results["vote_result_summary"] = compute_vote_summary(updated_results, votes_by_item_id)
        room.results_json = updated_results
        room.status = "voting_complete"
        event_type = "voting_complete"
    else:
        room.status = "voting_open"
        event_type = "vote_updated"

    bump_room(room)
    db.commit()
    db.refresh(room)

    await manager.broadcast(
        room.code,
        event_type=event_type,
        state_revision=room.state_revision,
        changed_sections=changed_sections,
    )

    return {"room": serialize_room(db, room, user)}


@app.websocket("/ws/rooms/{code}")
async def room_websocket(code: str, websocket: WebSocket, db: Session = Depends(get_db)) -> None:
    """
    Room-scoped WebSocket.

    The payloads are lightweight revision events.
    Complex rendering data is fetched over normal HTTP API calls.
    """
    user_id = websocket.session.get("user_id") if hasattr(websocket, "session") else None
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user = db.get(User, int(user_id))
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if not room or not is_room_member(db, room.id, user.id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(room.code, websocket)

    try:
        await websocket.send_json(
            {
                "event_type": "room_resync_required",
                "room_code": room.code,
                "state_revision": room.state_revision,
                "changed_sections": ["all"],
                "server_timestamp": utcnow().isoformat(),
            }
        )

        while True:
            # We do not need client messages for V1.
            # Receiving keeps the connection open and detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(room.code, websocket)
    except Exception:
        manager.disconnect(room.code, websocket)
        await websocket.close()
