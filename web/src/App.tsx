import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  CheckCircle2,
  Crown,
  Loader2,
  LogOut,
  Plus,
  RefreshCw,
  Sparkles,
  Users,
  Vote,
} from 'lucide-react'
import { API_BASE, apiFetch, assetUrl } from './api'

type User = {
  id: number
  email: string
  display_name: string
}

type RoomListItem = {
  code: string
  title: string
  status: string
  state_revision: number
  last_activity_at: string
}

type Participant = {
  user_id: number
  display_name: string
  is_host: boolean
  has_submitted: boolean
  has_voted: boolean
}

type AnimeItem = {
  catalog_item_id: number
  title: string
  media_type: string
  year: number
  status: string
  tags: string[]
  thumbnail_local_path: string
  image_local_path: string
  score?: number
  group_match_score?: number
  vote_count?: number
  is_winner?: boolean
}

type Cluster = {
  cluster_index: number
  cluster_label: string
  cluster_score: number
  top_items: AnimeItem[]
}

type RoomPayload = {
  code: string
  title: string
  status: string
  state_revision: number
  is_host: boolean
  constraints: {
    hard_constraint_year_start: number
    hard_constraint_year_end: number
    hard_constraint_allowed_types: string[]
  }
  participants: Participant[]
  own_submission: string
  own_vote_catalog_item_ids: number[]
  vote_progress: {
    voted_count: number
    member_count: number
    pending_count: number
  }
  results: null | {
    chosen_k: number
    kmeans_silhouette: number
    eligible_catalog_subset_size: number
    clusters: Cluster[]
    final_recommendations: AnimeItem[]
    vote_result_summary: AnimeItem[]
  }
}

function App() {
  const [user, setUser] = useState<User | null | undefined>(undefined)

  const refreshMe = useCallback(async () => {
    try {
      const data = await apiFetch<{ user: User }>('/api/auth/me')
      setUser(data.user)
    } catch {
      setUser(null)
    }
  }, [])

  useEffect(() => {
    refreshMe()
  }, [refreshMe])

  if (user === undefined) {
    return <LoadingScreen />
  }

  return (
    <Shell user={user} refreshMe={refreshMe}>
      <Routes>
        <Route path="/" element={user ? <Navigate to="/dashboard" /> : <Navigate to="/login" />} />
        <Route path="/login" element={<AuthPage mode="login" refreshMe={refreshMe} />} />
        <Route path="/register" element={<AuthPage mode="register" refreshMe={refreshMe} />} />
        <Route path="/dashboard" element={user ? <Dashboard /> : <Navigate to="/login" />} />
        <Route path="/rooms/new" element={user ? <CreateRoom /> : <Navigate to="/login" />} />
        <Route path="/rooms/join" element={user ? <JoinRoom /> : <Navigate to="/login" />} />
        <Route path="/rooms/:code" element={user ? <RoomPage /> : <Navigate to="/login" />} />
      </Routes>
    </Shell>
  )
}

function Shell({
  user,
  refreshMe,
  children,
}: {
  user: User | null
  refreshMe: () => Promise<void>
  children: React.ReactNode
}) {
  const navigate = useNavigate()

  async function logout() {
    await apiFetch('/api/auth/logout', { method: 'POST', body: '{}' })
    await refreshMe()
    navigate('/login')
  }

  return (
    <main className="min-h-screen bg-[#060816] text-slate-50">
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="bg-blob-a absolute left-[-10%] top-[-20%] h-[420px] w-[420px] rounded-full bg-fuchsia-500/30 blur-3xl" />
        <div className="bg-blob-b absolute bottom-[-20%] right-[-10%] h-[520px] w-[520px] rounded-full bg-cyan-500/20 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.10),transparent_30%),linear-gradient(180deg,rgba(6,8,22,0.2),#060816)]" />
      </div>

      <nav className="sticky top-0 z-20 border-b border-white/10 bg-slate-950/55 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <Link to="/dashboard" className="group flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-fuchsia-500 to-cyan-400 shadow-lg shadow-fuchsia-500/20 transition group-hover:scale-105">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <p className="text-lg font-black tracking-tight">AniSync</p>
              <p className="text-xs text-slate-400">Private anime group decisions</p>
            </div>
          </Link>

          {user && (
            <div className="flex items-center gap-3">
              <span className="hidden rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 sm:inline">
                {user.display_name}
              </span>
              <button
                onClick={logout}
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:-translate-y-0.5 hover:bg-white/10"
              >
                <span className="inline-flex items-center gap-2">
                  <LogOut className="h-4 w-4" />
                  Logout
                </span>
              </button>
            </div>
          )}
        </div>
      </nav>

      <div className="mx-auto max-w-6xl px-4 py-8">{children}</div>
    </main>
  )
}

function LoadingScreen() {
  return (
    <div className="grid min-h-screen place-items-center bg-[#060816] text-white">
      <div className="flex items-center gap-3 rounded-3xl border border-white/10 bg-white/5 px-6 py-4 backdrop-blur-xl">
        <Loader2 className="h-5 w-5 animate-spin text-cyan-300" />
        Loading AniSync...
      </div>
    </div>
  )
}

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className={`rounded-[2rem] border border-white/10 bg-white/[0.07] p-6 shadow-2xl shadow-black/20 backdrop-blur-2xl ${className}`}
    >
      {children}
    </motion.div>
  )
}

function PrimaryButton({
  children,
  disabled,
  type = 'button',
  onClick,
}: {
  children: React.ReactNode
  disabled?: boolean
  type?: 'button' | 'submit'
  onClick?: () => void
}) {
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className="group inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-fuchsia-500 to-cyan-400 px-5 py-3 font-bold text-white shadow-lg shadow-fuchsia-500/20 transition hover:-translate-y-0.5 hover:shadow-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
    >
      {children}
      <ArrowRight className="h-4 w-4 transition group-hover:translate-x-1" />
    </button>
  )
}

function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-slate-50 outline-none ring-0 transition placeholder:text-slate-500 focus:border-cyan-300/60 focus:bg-slate-950/80 ${props.className ?? ''}`}
    />
  )
}

function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={`min-h-32 w-full resize-y rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-slate-50 outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60 focus:bg-slate-950/80 ${props.className ?? ''}`}
    />
  )
}

function ErrorMessage({ message }: { message: string }) {
  if (!message) return null
  return (
    <div className="rounded-2xl border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-100">
      {message}
    </div>
  )
}

function AuthPage({ mode, refreshMe }: { mode: 'login' | 'register'; refreshMe: () => Promise<void> }) {
  const navigate = useNavigate()
  const [email, setEmail] = useState(mode === 'login' ? 'host@example.com' : '')
  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState(mode === 'login' ? 'AniSyncDemo123!' : '')
  const [error, setError] = useState('')
  const isRegister = mode === 'register'

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')

    try {
      await apiFetch(`/api/auth/${isRegister ? 'register' : 'login'}`, {
        method: 'POST',
        body: JSON.stringify({
          email,
          display_name: displayName,
          password,
        }),
      })
      await refreshMe()
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed.')
    }
  }

  return (
    <div className="mx-auto grid max-w-5xl items-center gap-8 py-12 md:grid-cols-[1.1fr_0.9fr]">
      <section>
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 inline-flex rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100"
        >
          Semantic anime recommendations for friend groups
        </motion.p>
        <h1 className="text-5xl font-black tracking-tight md:text-7xl">
          Stop scrolling.
          <span className="block bg-gradient-to-r from-fuchsia-300 via-white to-cyan-200 bg-clip-text text-transparent">
            Pick together.
          </span>
        </h1>
        <p className="mt-6 max-w-xl text-lg leading-8 text-slate-300">
          Create a private room, let everyone describe what they want to watch, then AniSync clusters
          anime choices into a clear final voting list.
        </p>
      </section>

      <Card>
        <h2 className="text-2xl font-black">{isRegister ? 'Create an account' : 'Login'}</h2>
        <p className="mt-2 text-sm text-slate-400">
          {isRegister ? 'Start a new private anime room.' : 'Use a seeded demo account or your own account.'}
        </p>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />

          {isRegister && (
            <Input
              placeholder="Display name"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              required
            />
          )}

          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />

          <ErrorMessage message={error} />

          <PrimaryButton type="submit">{isRegister ? 'Register' : 'Login'}</PrimaryButton>
        </form>

        <p className="mt-5 text-sm text-slate-400">
          {isRegister ? 'Already have an account?' : 'Need an account?'}{' '}
          <Link className="font-bold text-cyan-200 hover:text-cyan-100" to={isRegister ? '/login' : '/register'}>
            {isRegister ? 'Login' : 'Register'}
          </Link>
        </p>
      </Card>
    </div>
  )
}

function Dashboard() {
  const [rooms, setRooms] = useState<RoomListItem[]>([])
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const data = await apiFetch<{ rooms: RoomListItem[] }>('/api/rooms')
      setRooms(data.rooms)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load rooms.')
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return (
    <div className="space-y-8">
      <section className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <p className="mb-3 inline-flex rounded-full border border-fuchsia-300/20 bg-fuchsia-300/10 px-4 py-2 text-sm text-fuchsia-100">
            Dashboard
          </p>
          <h1 className="text-4xl font-black tracking-tight md:text-6xl">AniSync</h1>
          <p className="mt-3 text-slate-300">Private anime group recommendation for small groups.</p>
        </div>

        <div className="flex flex-wrap gap-3">
          <Link to="/rooms/new">
            <PrimaryButton>
              <Plus className="h-4 w-4" />
              Create Anime Room
            </PrimaryButton>
          </Link>
          <Link
            to="/rooms/join"
            className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 font-bold text-slate-100 transition hover:-translate-y-0.5 hover:bg-white/10"
          >
            Join Room by Code
          </Link>
        </div>
      </section>

      <ErrorMessage message={error} />

      <div className="grid gap-4 md:grid-cols-2">
        {rooms.map((room) => (
          <Link key={room.code} to={`/rooms/${room.code}`}>
            <Card className="group transition hover:-translate-y-1 hover:border-cyan-300/30">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-black">{room.title}</h2>
                  <p className="mt-1 text-sm text-slate-400">Room code: {room.code}</p>
                </div>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs uppercase tracking-widest text-cyan-100">
                  {room.status}
                </span>
              </div>
            </Card>
          </Link>
        ))}

        {rooms.length === 0 && (
          <Card>
            <p className="text-slate-300">No rooms yet. Create a room to start the demo flow.</p>
          </Card>
        )}
      </div>
    </div>
  )
}

function CreateRoom() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('Friday Anime Night')
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')

    try {
      const data = await apiFetch<{ room: RoomPayload }>('/api/rooms', {
        method: 'POST',
        body: JSON.stringify({ title }),
      })
      navigate(`/rooms/${data.room.code}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create room.')
    }
  }

  return (
    <Card className="mx-auto max-w-xl">
      <h1 className="text-3xl font-black">Create an Anime Room</h1>
      <p className="mt-2 text-slate-400">Create a private invite-only room for your group.</p>

      <form onSubmit={submit} className="mt-6 space-y-4">
        <Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Room title" required />
        <ErrorMessage message={error} />
        <PrimaryButton type="submit">Create Room</PrimaryButton>
      </form>
    </Card>
  )
}

function JoinRoom() {
  const navigate = useNavigate()
  const [code, setCode] = useState('')
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')

    try {
      const data = await apiFetch<{ room: RoomPayload }>('/api/rooms/join', {
        method: 'POST',
        body: JSON.stringify({ code }),
      })
      navigate(`/rooms/${data.room.code}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not join room.')
    }
  }

  return (
    <Card className="mx-auto max-w-xl">
      <h1 className="text-3xl font-black">Join Room</h1>
      <p className="mt-2 text-slate-400">Enter the room code your friend shared.</p>

      <form onSubmit={submit} className="mt-6 space-y-4">
        <Input value={code} onChange={(event) => setCode(event.target.value.toUpperCase())} placeholder="Room code" required />
        <ErrorMessage message={error} />
        <PrimaryButton type="submit">Join Room</PrimaryButton>
      </form>
    </Card>
  )
}

function RoomPage() {
  const { code = '' } = useParams()
  const [room, setRoom] = useState<RoomPayload | null>(null)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('connecting')
  const lastRevisionRef = useRef(0)

  const refreshRoom = useCallback(async () => {
    const data = await apiFetch<{ room: RoomPayload }>(`/api/rooms/${code}`)
    setRoom(data.room)
    lastRevisionRef.current = data.room.state_revision
  }, [code])

  useEffect(() => {
    refreshRoom().catch((err) => setError(err instanceof Error ? err.message : 'Could not load room.'))
  }, [refreshRoom])

  useEffect(() => {
    let stopped = false
    let socket: WebSocket | null = null
    let retry = 0

    function connect() {
      const wsBase = API_BASE.replace(/^http/, 'ws')
      socket = new WebSocket(`${wsBase}/ws/rooms/${code}`)

      socket.onopen = () => {
        retry = 0
        setConnectionState('live')
      }

      socket.onmessage = async (event) => {
        const payload = JSON.parse(event.data)
        if (payload.state_revision > lastRevisionRef.current || payload.changed_sections?.includes('all')) {
          await refreshRoom()
        }
      }

      socket.onclose = () => {
        if (stopped) return
        setConnectionState('reconnecting')
        retry += 1
        window.setTimeout(connect, Math.min(1000 * 2 ** retry, 10000))
      }
    }

    connect()

    return () => {
      stopped = true
      socket?.close()
    }
  }, [code, refreshRoom])

  if (!room) {
    return (
      <Card>
        <Loader2 className="mr-2 inline h-5 w-5 animate-spin" />
        Loading room...
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <RoomHeader room={room} connectionState={connectionState} refreshRoom={refreshRoom} />
      <ErrorMessage message={error} />

      <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="space-y-6">
          <ParticipantsCard room={room} />
          <ConstraintsCard room={room} refreshRoom={refreshRoom} />
          <PreferenceCard room={room} refreshRoom={refreshRoom} />
        </div>

        <div className="space-y-6">
          {room.is_host && <ComputeCard room={room} refreshRoom={refreshRoom} />}
          {room.results ? <ResultsCard room={room} refreshRoom={refreshRoom} /> : <WaitingCard />}
        </div>
      </div>
    </div>
  )
}

function RoomHeader({
  room,
  connectionState,
  refreshRoom,
}: {
  room: RoomPayload
  connectionState: string
  refreshRoom: () => Promise<void>
}) {
  return (
    <Card>
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
        <div>
          <p className="mb-3 inline-flex rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100">
            Anime room · {room.code}
          </p>
          <h1 className="text-4xl font-black tracking-tight">{room.title}</h1>
          <p className="mt-2 text-slate-400">
            Status: <span className="font-bold text-slate-200">{room.status}</span> · Revision {room.state_revision}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm">
            {connectionState === 'live' ? '🟢 Live sync on' : '🟡 Reconnecting'}
          </span>
          <button
            onClick={() => refreshRoom()}
            className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm transition hover:bg-white/10"
          >
            <RefreshCw className="mr-2 inline h-4 w-4" />
            Refresh Room State
          </button>
        </div>
      </div>
    </Card>
  )
}

function ParticipantsCard({ room }: { room: RoomPayload }) {
  return (
    <Card>
      <h2 className="flex items-center gap-2 text-xl font-black">
        <Users className="h-5 w-5 text-cyan-200" />
        Participants
      </h2>

      <div className="mt-4 space-y-3">
        {room.participants.map((participant) => (
          <div key={participant.user_id} className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-3">
            <div className="flex items-center gap-3">
              <div className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-fuchsia-400/80 to-cyan-300/80 font-black">
                {participant.display_name.slice(0, 1).toUpperCase()}
              </div>
              <div>
                <p className="font-bold">
                  {participant.display_name}
                  {participant.is_host && <Crown className="ml-2 inline h-4 w-4 text-yellow-300" />}
                </p>
                <p className="text-xs text-slate-400">
                  {participant.has_submitted ? 'Preference submitted' : 'Waiting for preference'} ·{' '}
                  {participant.has_voted ? 'Voted' : 'Not voted'}
                </p>
              </div>
            </div>
            {participant.has_submitted && <CheckCircle2 className="h-5 w-5 text-emerald-300" />}
          </div>
        ))}
      </div>
    </Card>
  )
}

function ConstraintsCard({ room, refreshRoom }: { room: RoomPayload; refreshRoom: () => Promise<void> }) {
  const [startYear, setStartYear] = useState(room.constraints.hard_constraint_year_start)
  const [endYear, setEndYear] = useState(room.constraints.hard_constraint_year_end)
  const [allowedTypes, setAllowedTypes] = useState<string[]>(room.constraints.hard_constraint_allowed_types)
  const [error, setError] = useState('')

  useEffect(() => {
    setStartYear(room.constraints.hard_constraint_year_start)
    setEndYear(room.constraints.hard_constraint_year_end)
    setAllowedTypes(room.constraints.hard_constraint_allowed_types)
  }, [room.constraints])

  async function save(reset = false) {
    setError('')
    try {
      await apiFetch(`/api/rooms/${room.code}/constraints`, {
        method: 'POST',
        body: JSON.stringify({
          hard_constraint_year_start: startYear,
          hard_constraint_year_end: endYear,
          hard_constraint_allowed_types: allowedTypes,
          reset_to_defaults: reset,
        }),
      })
      await refreshRoom()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save constraints.')
    }
  }

  const allTypes = ['TV', 'MOVIE', 'OVA', 'ONA', 'SPECIAL']

  return (
    <Card>
      <h2 className="text-xl font-black">Room Hard Constraints</h2>
      <p className="mt-2 text-sm text-slate-400">
        Only anime that satisfy these filters can be used anywhere in this room session.
      </p>

      {room.is_host ? (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <Input type="number" value={startYear} onChange={(event) => setStartYear(Number(event.target.value))} />
            <Input type="number" value={endYear} onChange={(event) => setEndYear(Number(event.target.value))} />
          </div>

          <div className="flex flex-wrap gap-2">
            {allTypes.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => {
                  setAllowedTypes((current) =>
                    current.includes(type) ? current.filter((item) => item !== type) : [...current, type],
                  )
                }}
                className={`rounded-full border px-4 py-2 text-sm font-bold transition ${
                  allowedTypes.includes(type)
                    ? 'border-cyan-300/60 bg-cyan-300/20 text-cyan-50'
                    : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10'
                }`}
              >
                {type}
              </button>
            ))}
          </div>

          <ErrorMessage message={error} />

          <div className="flex flex-wrap gap-3">
            <PrimaryButton onClick={() => save(false)}>Save Hard Constraints</PrimaryButton>
            <button
              onClick={() => save(true)}
              className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 font-bold transition hover:bg-white/10"
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
          <p>Allowed release years: {startYear}–{endYear}</p>
          <p>Allowed types: {allowedTypes.join(', ')}</p>
        </div>
      )}
    </Card>
  )
}

function PreferenceCard({ room, refreshRoom }: { room: RoomPayload; refreshRoom: () => Promise<void> }) {
  const [queryText, setQueryText] = useState(room.own_submission)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setQueryText(room.own_submission)
  }, [room.own_submission])

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setSaved(false)

    try {
      await apiFetch(`/api/rooms/${room.code}/submit`, {
        method: 'POST',
        body: JSON.stringify({ query_text: queryText }),
      })
      setSaved(true)
      await refreshRoom()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save preference.')
    }
  }

  return (
    <Card>
      <h2 className="text-xl font-black">Describe what you want to watch</h2>
      <p className="mt-2 text-sm text-slate-400">
        Example: I want dark psychological anime with mystery and high-stakes action.
      </p>

      <form onSubmit={submit} className="mt-4 space-y-4">
        <Textarea value={queryText} onChange={(event) => setQueryText(event.target.value)} />
        <ErrorMessage message={error} />
        {saved && <p className="text-sm text-emerald-200">Saved. Other room pages will update automatically.</p>}
        <PrimaryButton type="submit">Save / Update Preference</PrimaryButton>
      </form>
    </Card>
  )
}

function ComputeCard({ room, refreshRoom }: { room: RoomPayload; refreshRoom: () => Promise<void> }) {
  const [error, setError] = useState('')
  const [computing, setComputing] = useState(false)
  const submittedCount = room.participants.filter((p) => p.has_submitted).length

  async function compute() {
    setError('')
    setComputing(true)

    try {
      await apiFetch(`/api/rooms/${room.code}/compute`, {
        method: 'POST',
        body: '{}',
      })
      await refreshRoom()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not compute recommendations.')
    } finally {
      setComputing(false)
    }
  }

  return (
    <Card>
      <h2 className="text-xl font-black">Host Controls</h2>
      <p className="mt-2 text-sm text-slate-400">
        {submittedCount} participant(s) have submitted. At least 2 are required.
      </p>

      <div className="mt-4">
        <PrimaryButton disabled={computing || submittedCount < 2} onClick={compute}>
          {computing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Generate Group Recommendations
        </PrimaryButton>
      </div>

      <ErrorMessage message={error} />
    </Card>
  )
}

function WaitingCard() {
  return (
    <Card>
      <h2 className="text-xl font-black">Recommendations will appear here</h2>
      <p className="mt-2 text-slate-400">
        After the host computes, this area will show clustered anime lists, the final voting list, and the group result.
      </p>
    </Card>
  )
}

function ResultsCard({ room, refreshRoom }: { room: RoomPayload; refreshRoom: () => Promise<void> }) {
  const results = room.results
  const [selectedVotes, setSelectedVotes] = useState<number[]>(room.own_vote_catalog_item_ids)
  const [error, setError] = useState('')

  useEffect(() => {
    setSelectedVotes(room.own_vote_catalog_item_ids)
  }, [room.own_vote_catalog_item_ids])

  if (!results) return null

  async function submitVotes() {
    setError('')
    try {
      await apiFetch(`/api/rooms/${room.code}/vote`, {
        method: 'POST',
        body: JSON.stringify({ catalog_item_ids: selectedVotes }),
      })
      await refreshRoom()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not submit votes.')
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="text-2xl font-black">Clustered Anime Lists</h2>
        <p className="mt-2 text-sm text-slate-400">
          K={results.chosen_k}, silhouette={results.kmeans_silhouette}. Each cluster is ranked by Group Match Score.
        </p>

        <div className="mt-5 space-y-5">
          {results.clusters.map((cluster, index) => (
            <div key={cluster.cluster_index} className="rounded-[1.5rem] border border-white/10 bg-slate-950/40 p-4">
              <h3 className="font-black">
                Cluster {index + 1}
                {cluster.cluster_label ? <span className="text-cyan-200"> — {cluster.cluster_label}</span> : null}
              </h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {cluster.top_items.map((item) => (
                  <AnimeCard key={item.catalog_item_id} item={item} compact />
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <h2 className="flex items-center gap-2 text-2xl font-black">
          <Vote className="h-6 w-6 text-fuchsia-200" />
          Final Recommended Anime List
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          Select one or more anime. Vote counts stay hidden until everyone has voted.
        </p>

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {results.final_recommendations.map((item) => {
            const checked = selectedVotes.includes(item.catalog_item_id)
            return (
              <button
                key={item.catalog_item_id}
                onClick={() => {
                  setSelectedVotes((current) =>
                    checked
                      ? current.filter((id) => id !== item.catalog_item_id)
                      : [...current, item.catalog_item_id],
                  )
                }}
                className={`rounded-[1.5rem] border p-2 text-left transition hover:-translate-y-0.5 ${
                  checked ? 'border-cyan-300/60 bg-cyan-300/10' : 'border-white/10 bg-white/[0.04]'
                }`}
              >
                <AnimeCard item={item} compact />
              </button>
            )
          })}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <PrimaryButton onClick={submitVotes}>
            {room.own_vote_catalog_item_ids.length ? 'Update Votes' : 'Submit Votes'}
          </PrimaryButton>
          <span className="text-sm text-slate-400">
            {room.vote_progress.voted_count} of {room.vote_progress.member_count} members have voted.
          </span>
        </div>

        <ErrorMessage message={error} />
      </Card>

      <AnimatePresence>
        {room.status === 'voting_complete' && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <Card className="border-cyan-300/30">
              <h2 className="text-3xl font-black">Final Group Result</h2>
              <p className="mt-2 text-slate-400">Sorted by vote count, group match score, then title.</p>

              <div className="mt-5 space-y-3">
                {results.vote_result_summary.map((item) => (
                  <div
                    key={item.catalog_item_id}
                    className={`rounded-[1.5rem] border p-3 ${
                      item.is_winner ? 'border-yellow-300/50 bg-yellow-300/10' : 'border-white/10 bg-white/[0.04]'
                    }`}
                  >
                    <AnimeCard item={item} compact showVotes />
                  </div>
                ))}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function AnimeCard({ item, compact, showVotes }: { item: AnimeItem; compact?: boolean; showVotes?: boolean }) {
  return (
    <div className="flex gap-3">
      <img
        src={assetUrl(item.thumbnail_local_path || item.image_local_path)}
        alt={item.title}
        className={`${compact ? 'h-28 w-20' : 'h-40 w-28'} rounded-2xl object-cover shadow-lg shadow-black/30`}
        loading="lazy"
      />
      <div className="min-w-0 flex-1">
        <h4 className="line-clamp-2 font-black">{item.title}</h4>
        <p className="mt-1 text-xs uppercase tracking-widest text-slate-400">
          {item.media_type} · {item.year} · {item.status}
        </p>
        <div className="mt-2 flex flex-wrap gap-1">
          {(item.tags ?? []).slice(0, 3).map((tag) => (
            <span key={tag} className="rounded-full bg-white/10 px-2 py-1 text-[11px] text-slate-300">
              {tag}
            </span>
          ))}
        </div>
        {item.group_match_score !== undefined && (
          <p className="mt-2 text-sm font-bold text-cyan-100">
            Group Match Score: {item.group_match_score.toFixed(4)}
          </p>
        )}
        {showVotes && (
          <p className="mt-1 text-sm font-bold text-yellow-100">
            Votes: {item.vote_count} {item.is_winner ? '· Winner' : ''}
          </p>
        )}
      </div>
    </div>
  )
}

export default App
