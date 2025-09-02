using Crayons, Random, DataStructures



Pos = Tuple{Int, Int}
Base.:+(a::Pos, b::Pos) = (a[1]+b[1], a[2]+b[2])
Base.zero(::Type{Pos}) = (0, 0)
Base.isequal(a::Pos, b::Pos) = (a[1] == b[1]) && (a[2] == b[2])
Base.:!=(a::Pos, b::Pos) = (a[1] != b[1]) || (a[2] != b[2])
in_boundary(p::Pos) = (p[1] >= 1 && p[1] <= 5 && p[2]>=1 && p[2]<=5)

# const dx::Vector{Int} = [-1, -1, -1, 0, 1, 1, 1, 0]
# const dy::Vector{Int} = [-1, 0, 1, 1, 1, 0, -1, -1]

const dpos::Vector{Pos} = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
const dpos_char = ['↖','↑','↗','→','↘','↓','↙','←']
const CHAR = [["$(Crayon(foreground=:blue)("♠"))", "$(Crayon(foreground=:blue, bold=true)("♣"))"],
    ["$(Crayon(foreground=:red)("♥"))", "$(Crayon(foreground=:red)("♦"))"]]

const dcell = [-6, -1, +4, 5, 6, 1, -4, -5]
const xy2cell(x, y) = x+y*5-5
const cell2xy(cell) = mod(cell-1, 5)+1, div(cell-1, 5)+1
const NEIGHBORS = [
    [cell + x for x in dcell if 1 <= cell + x <= 25 && abs( cell2xy(cell)[1] - cell2xy(cell+x)[1] ) <= 1 && abs( cell2xy(cell)[2] - cell2xy(cell+x)[2] ) <= 1 ] for cell in 1:25
]

const INITIALIZATION = -1
const ACTION_MOVE = 0
const ACTION_BUILD = 1
const ACTION_SPECIAL = 2


mutable struct SantoriniMove <: AbstractMove
    idWorker::Int  #1 or 2
    cell::Int
    kind::Int   # ACTION: MOVE, BUILD or SPECIAL
end

Base.:(==)(m::SantoriniMove, n::SantoriniMove) = (m.idWorker == n.idWorker) && (m.cell == n.cell) && (m.kind == n.kind)   #necessary for comparing moves in keep_subtree! for instance

Base.zero(::Type{SantoriniMove}) = SantoriniMove(0, 0, 0)

const GODNAMES = [
    "No god",
    "Apollo",
    "Artemis",
    "Athena",
    "Atlas",
    "Demeter",
    "Hephaestus",
    "Hermes",
    "Minotaur",
    "Pan",
    "Prometheus"
]

# Initialization
# just use void filters
# Notes on gods : 
# 1 Appollo : move to enemy cell swaps him to where you just came from    -   no code change
# 2 Artemis : move once or twice                          -  possible : remaining moves filters - range filters
# 3 Athena : if you move up, opponent cant move up        - range filters
# 4 Atlas : dome on levels 0, 1 and 2 aswell              - no code change
# 5 Demeter : build once or twice (not on same building)  - remaning build filter - TODO also add double build action ? or pass action ?
# 6 Hephaestus : build twice on same building             - remaining build filter
# 7 Hermes : infinite move as long as he does not moves up or down    - range filters OK
# 8 Minotaur : bumps when moving to a cell if the one behind is free  - range filters OK
# 9 Pan : 2 to 0 wins - no code change
# 10 Prometheus : build before and after moving if moving is not up - range filters + remaining build OK
# 11 Aphrodite : if opponent starts near you - must finish near you - range filters
# 12 Ares : remove any unoccupied 1-2-3 level - TODO : additional action ?
# 13 Bia : cellx -> celly implies that opponent on celly+1 is removed from game - no code change except win condition
# 14 Chaos : use power of gods 1-10 randomly  -  set easily in params
# 15 Charon : before action : takes an opponent and moves its place - TODO : additional action
# 16 Chronus : win if 5 completed towers - no code change
# 17 Circe : you use their god power if they are not glued - easily set in params
# 18 Dionysus : if you complete a tower => make another turn with their workers - possible? switch wk1 wk2 in inputs
# 19 Eros : win condition changes
# 20 Hera : win condition changes
# 21 Hestia : build twice but not on perimeter OK
# 22 Hypnus : opponent worker is highest => cant move
# 23 Limus : opponent can't build around you (except 4)
# 24 Medusa : if you can build on lower opponent : they dead
# 25 Morpheus : dont
# 26 Persephone : opponents must move up
# 27 Poseidon : unmoved 
# https://cdn.1j1ju.com/medias/fc/ec/5d-santorini-rulebook.pdf
# NN inputs : field + worker1 + worker2 + opponentworkers + (remainingmove?) + (remainingbuild?)  // params : gods me + gods he
# action output : move wk1 - move wk2 - build wk1 - build wk2 - additional action wk1 - additional action wk2

const NOGOD = 1
const APOLLO = 2    # move into enemy actually swaps
const ARTEMIS = 3   # move from 1 or 2 cells
const ATHENA = 4    # if moves up, opponent cant move up
const ATLAS = 5   # can build domes
const DEMETER = 6   # build once or twice, not on same building - optional !
const HEPHAESTUS = 7    # build once or twice on same building
const HERMES = 8        # 0 to infinite moves if not going up or down, for the two workers
const MINOTAUR = 9      # bumps
const PAN = 10       # 0->2 wins
const PROMETHEUS = 11    # builds (optional) -> moves (not up if has built) -> builds
const NBGODS = 11

# Santorini
mutable struct Santorini <: AbstractGame
    board::Vector{Int}  #levels on cells 1 to 25 ; board[cell]
    workers::Array{Int, 2}  #worker[player][id (1 or 2)]
    gods::Vector{Int}   #gods[p] 
    buildWorker::Int    # 0 if no worker must build ; 1 if worker 1 must build ; 2 if worker 2 must build
    winner::Int
    playerToPlay::Int
    isFinished::Bool
    godFlag::Vector{Int}       #= used for : athena : a godflag set to 1 means that opponent cant move up
                                            demeter : a godflag set to X means that builder builds a second time, but not on X
                                            hermes : a godflag set to 1 means that 3-G.buildWorker can also move and only on equal levels
                                            prometheus : godflag to 1 or 2 means that this worker built before moving
                                =#

    function Santorini()
        board = zeros(Int, 25)
        workers = zeros(Int, 2, 2)
        gods = [1, 1]
        if rand(1:10) != 1
            gods[1] = gods[2] = rand(2:NBGODS)
            while gods[2] == gods[1]
                gods[2] = rand(2:NBGODS)
            end
        end
        return new(board, workers, gods, INITIALIZATION, -1, 1, false, [0, 0])
    end
end

function Game.get_null_move(::Santorini) :: SantoriniMove
    return SantoriniMove(0, 0, 0)
end

function Game.number_played_moves(G::Santorini) :: Int
    return sum(G.board)
end

function Game.is_finished(G::Santorini) :: Bool
    return G.isFinished
end

function Game.get_winner(G::Santorini) :: Int
    return G.winner
end

function Game.move_to_string(::Santorini, move::SantoriniMove) :: String
    return string(move.idWorker)*" "*string(move.cell)*" "*string(move.kind)
end
function Game.string_to_move(::Santorini, s::String) :: SantoriniMove
    id, cell, kind = map(x -> parse(Int, x), split(s, ' '))
    return SantoriniMove(id, cell, kind)
end

function Game.get_printable_move(G::Santorini, move::SantoriniMove) :: String
    text = CHAR[G.playerToPlay][move.idWorker]*" "
    if move.kind == ACTION_MOVE
        text *= " moves to "
    elseif move.kind == ACTION_BUILD
        text *= " builds on "
    elseif G.gods[G.playerToPlay] == ATLAS
        text *= " builds dome on "
    elseif G.gods[G.playerToPlay] == DEMETER
        text *= " does nothing else"
        return text
    elseif G.gods[G.playerToPlay] == HEPHAESTUS
        text *= " builds double on "
    end
    text *= string(cell2xy(move.cell)[1])*" "*string(cell2xy(move.cell)[2])
    return text
end


function Base.print(G::Santorini)
    println("GOD1 : ", GODNAMES[G.gods[1]], "\tGOD2: ", GODNAMES[G.gods[2]])
    println(G.godFlag, " ", G.buildWorker)
    print(" ")
    for y in 1:5
        print(" ", y, " ")
    end
    println("\n  ---------------")

    #color = [[crayon"light_red", crayon"red"], [crayon"light_blue", crayon"blue"]]
    #symbol = ['o', 'X']
    levels = [' ', '▬', '▄', '█', '◓']
    for x in 1:5
        print(x, "|")
        for y in 1:5
            occupied = false
            cell = xy2cell(x, y)
            for p in 1:2, i in 1:2
                if  G.workers[p, i] == cell
                    print(CHAR[p][i])
                    occupied = true
                end
            end
            if !occupied
                print(" ")
            end
            print(levels[G.board[cell]+1], "|")
        end
        println("\n  ---------------")
    end
    if G.isFinished
        println("Winner : ", CHAR[G.winner][1], "/", CHAR[G.winner][2])
    elseif G.buildWorker == INITIALIZATION
        println(CHAR[G.playerToPlay][1], "/", CHAR[G.playerToPlay][2], " must be placed ")
    elseif G.buildWorker == 0
        println(CHAR[G.playerToPlay][1], "/", CHAR[G.playerToPlay][2], " moves ")
    else
        println(CHAR[G.playerToPlay][G.buildWorker], " builds")
    end
end