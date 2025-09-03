const NUM_PLAYERS = 2
const ACTION_RANDOM = 0
const ACTION_TAKE = 1
const ACTION_PLACE = 2
const FLOOR_LINE = 6
const FACTORIES = 2*NUM_PLAYERS+1
const CENTER = FACTORIES + 1 # 6 at 2p, 8 at 3p, 10 at 4p

const AZUL = 1
const YELLOW = 2
const RED = 3
const BLACK = 4
const WHITE = 5
const CHAR = ["$(Crayon(foreground=:blue)("A"))", "$(Crayon(foreground=:yellow)("Y"))", "$(Crayon(foreground=:red)("R"))", "$(Crayon(foreground=:dark_gray)("B"))", "$(Crayon(foreground=:white)("W"))"]
const CRAYONSCOLORS = [crayon"blue", crayon"yellow", crayon"red", crayon"dark_gray", crayon"white"]
const BOARDCOLOR = [1 2 3 4 5 ; 5 1 2 3 4 ; 4 5 1 2 3 ; 3 4 5 1 2 ; 2 3 4 5 1]
const COLORPOS = [1 2 3 4 5 ; 2 3 4 5 1 ; 3 4 5 1 2 ; 4 5 1 2 3 ; 5 1 2 3 4]

#a move is divided into 2 moves : (0) : from a factory or center (6), (1) : toward a pattern line or floor line
mutable struct AzulMove <: AbstractMove
    kind::Int   # 0: factory 1: board
    line::Int # 1-5 : factory/pattern ; 6 : center/floor
    color::Int
    seed::Int   # for ACTION_RANDOM
end

function Base.:(==)(m::AzulMove, n::AzulMove)    #necessary for comparing moves in keep_subtree! for instance
    return (m.seed == 0 && n.seed == 0) ? m.kind == n.kind && m.line == n.line && m.color == n.color : m.seed == n.seed
end

function draw(rng, bag)
    a = rand(rng, 1:sum(bag))
    i = 1
    while a > bag[i]
        a -= bag[i]
        i += 1
    end
    return i
end


mutable struct Azul <: AbstractPointsGame
    board::Array{Bool, 3}               # board[x, y, p]
    pattern::Array{Tuple{Int, Int}, 2}  # pattern[p, line] is of form (qty, color)
    floor::Vector{Int}               # floor[1] floor[2]
    scores::Vector{Int}                 # scores[1] scores[2]
    hasFirstPlayerTile::Vector{Bool}

    factory::Array{Int, 2}  #factory[line 1 to 6, color 1 to 5]
    buffer::Vector{Int} #(qty, color) between taking and placing

    playerToPlay::Int
    finished::Bool
    round::Int

    bag::Vector{Int}    #remaining of colors 1 to 5

    core_rng::Random.Xoshiro

    function Azul()
        core_rng = Random.Xoshiro()
        bag = [20 for color in 1:5]
        factory = [0 for line in 1:CENTER, color in 1:5]

        for line in 1:FACTORIES, qty in 1:4
            color = draw(core_rng, bag)
            bag[color] -= 1
            factory[line, color] += 1
        end
        board = zeros(Bool, 5, 5, NUM_PLAYERS)
        return new(board, [(0,0) for p in 1:NUM_PLAYERS, color in 1:5], zeros(Int, NUM_PLAYERS), zeros(Int, NUM_PLAYERS), zeros(Bool, NUM_PLAYERS), factory, [0,0,0,0,0], 1, false, 1, bag, core_rng)
    end
    Azul(G::Azul) = new(copy(G.board), deepcopy(G.pattern), copy(G.floor), copy(G.scores), copy(G.hasFirstPlayerTile), copy(G.factory), copy(G.buffer), G.playerToPlay, G.finished, G.round, copy(G.bag), Random.Xoshiro(rand(G.core_rng, Int)))
end


function place_tile(board, line, color)
    x = line
    y = COLORPOS[line, color]
    # how many horizontal neighbor ?
    horizontal = 1
    yy = y-1
    while yy >= 1 && board[x, yy]
        yy -= 1
        horizontal += 1
    end
    yy = y+1
    while yy <= 5 && board[x, yy]
        yy += 1
        horizontal += 1
    end

    #how many vertical neighbor ?
    vertical = 1
    xx = x-1
    while xx >= 1 && board[xx, y]
        xx -= 1
        vertical += 1
    end
    xx = x+1
    while xx <= 5 && board[xx, y]
        xx += 1
        vertical += 1
    end

    #score
    if horizontal == 1 && vertical == 1
        return 1
    elseif horizontal > 1 && vertical > 1
        return horizontal + vertical
    else
        return horizontal + vertical - 1
    end
end

function floor_points(floor)
    if floor >= 7
        return 14
    else
        return [0, 1, 2, 4, 6, 8, 11][floor+1]
    end
end


function Game.is_finished(G::Azul) :: Bool
    return G.finished
end



function get_completed_rows(G::Azul, p::Int)
    return count(sum(G.board[:, :, p]; dims=2) .== 5)
end
function get_completed_cols(G::Azul, p::Int)
    return count(sum(G.board[:, :, p]; dims=1) .== 5)
end
function get_completed_colors(G::Azul, p::Int)
    colors = 0
    for color in 1:5
        if G.board[1, COLORPOS[1, color], p] && G.board[2, COLORPOS[2, color], p] && G.board[3, COLORPOS[3, color], p] && G.board[4, COLORPOS[4, color], p] && G.board[5, COLORPOS[5, color], p]
            colors += 1
        end
    end
    return colors
end

function Game.get_winner(G::Azul)
    #=if NUM_PLAYERS != 2
        ranks = zeros(NUM_PLAYERS)
        scores = [10*G.scores[p] + get_completed_rows(G, p) for p in 1:NUM_PLAYERS]
        r = sortperm(G.scores; rev=true)
        i=1
        while i <= length(r)
            j = i+1
            while j <= length(r) && scores[r[i]] == scores[r[j]]
                j += 1
            end
            #equality on [i j[
            ranks[r[i:j-1]] .= sum(i:j-1) / (j-i)
            i = j

        end
        return ranks
    end=#

    max_score = maximum(G.scores)
    if count(G.scores .== max_score) == 1
       for p in 1:NUM_PLAYERS
           if G.scores[p] == max_score
               return p
           end
       end
    elseif NUM_PLAYERS == 2
        horizontal = [get_completed_rows(G, p) for p in 1:2]
        if horizontal[1] > horizontal[2]
            return 1
        elseif horizontal[2] > horizontal[1]
            return 2
        else
            return 0
        end
    else
        max_score_b = maximum(10*G.scores + get_completed_rows(G, p))
        for p in 1:NUM_PLAYERS
           if 10*G.scores[p] + get_completed_rows(G, p) == max_score_b
               return p
           end
        end
        @assert false
    end
end


function Game.get_delta_points(G::Azul)
    #raw score, rows, cols, colors
    return [ [G.scores[p] - 2*get_completed_rows(G, p) - 7*get_completed_cols(G, p) - 10*get_completed_colors(G, p), get_completed_rows(G, p), get_completed_cols(G, p), get_completed_colors(G, p)] for p in 1:NUM_PLAYERS]
end
function get_raw_points_round(G::Azul)
    if !G.finished
        return G.scores
    else
        return [G.scores[p] - 2*get_completed_rows(G, p) - 7*get_completed_cols(G, p) - 10*get_completed_colors(G, p) for p in 1:NUM_PLAYERS]
    end
end

function Game.get_utility_score(::Azul, deltaScore) :: Float32
    return atan(deltaScore / 10) * 0.3165 + 0.5 # in [0, 1]
end

function Game.points_to_utility(G::Azul, points::Vector{Float32}) :: Float32
    P = points[1] + points[2] + 2*points[3] + 7*points[4] + 10*points[5]
    return get_utility_score(G, P)
end

function Game.number_played_moves(G::Azul) :: Int
    return 5*G.round - sum(G.factory) + 15     #at round 1, with 20 in factories : 0 ; at round 5, with 0 in factories : 40
#     return G.round
end

function Game.get_printable_move(G::Azul, move::AzulMove) :: String
    text = ""
    if move.kind == ACTION_RANDOM
        text = "random seed = "*string(move.seed)
    elseif move.kind == ACTION_TAKE
        text = "Take " * join( [CHAR[move.color] for i in 1:G.factory[move.line, move.color]] )
        text = text * " from "*(move.line == CENTER ? "center" : string(move.line))
    else
        text = "Place " * join( [CHAR[move.color] for i in 1:G.buffer[move.color]] )
        text = text * " to "*(move.line == FLOOR_LINE ? " floor " : string(move.line))
    end
#     text = (move.kind == ACTION_TAKE) ? "Take " : "Place "
#     text = text * CHAR[move.color]
#     if move.kind == ACTION_TAKE
#         text = text * " from "*(move.line == CENTER ? "center" : string(move.line))

#     else
#         text = text * " to "*(move.line == FLOOR_LINE ? " floor " : string(move.line))
#     end
    return text
end
function Game.move_to_string(G::Azul, move::AzulMove) :: String
    return string(move.kind)*" "*string(move.line)*" "*string(move.color)*" "*string(move.seed)
end
function Game.string_to_move(G::Azul, s::String)
    kind, line, color, seed = split(s, ' ')
    kind = parse(Int, kind)
    line = parse(Bool, line)
    color = parse(Int, color)
    seed = parse(Int, seed)
    return AzulMove(kind, line, color, seed)
end

function Game.get_null_move(G::Azul) :: AzulMove
    return AzulMove(-1, -1, -1, 0)
end


function text_pattern(pattern, line)
    qty, color = pattern[line]
    c = (qty == 0) ? "$(Crayon(reset=true)("."))" : CHAR[color]
    return  join(reverse( [[c for i in 1:qty]..., ['.' for i in (qty+1):line]..., [' ' for i in (line+1):6]...] ))
end
text_board(board, line) = join([ join([CRAYONSCOLORS[BOARDCOLOR[line, y]], (board[line, y] ? 'X' : '.')]) for y in 1:5 ])
function text_factory(factory)
    text = ""
    for color in 1:5
        for x in 1:factory[color]
            text = text * CHAR[color]
        end
    end
    if text == ""
        return "    "
    end
    return text
end

function Base.print(G::Azul)
    print("Player to play : ", G.playerToPlay, " buffer : ")
    println( join([    join([CHAR[color] for x in 1:G.buffer[color]])  for color in 1:5]) )

    println("Score : ", join(G.scores, "\t\t"))
    for line in 1:5
        for p in 1:NUM_PLAYERS
            print(text_pattern(G.pattern[p, :], line), ' ')
            print(text_board(G.board[:, :, p], line))
            print("\t")
        end
        println()
    end
    print(crayon"reset")
    for p in 1:NUM_PLAYERS
        hasFP = G.hasFirstPlayerTile[p]
        #print( hasFP ? "1" : "",  join([['X' for x in (1+hasFP):G.floor[p]]..., [ '.' for x in (G.floor[p]+1):7 ]...] ) )
        print( hasFP ? "1" : "",  join(['X' for x in (1+hasFP):min(7,G.floor[p])]), join([ '.' for x in (G.floor[p]+1):7 ]))
        print("\t\t")
    end
    println()

    println(join(["-" for x in 1:50]))
    for line in 1:FACTORIES
        print(" |", string(line), " : ", text_factory(G.factory[line, :]))
    end
    hasFP = !any(G.hasFirstPlayerTile)
    println("\n\t\tC ", (hasFP ? "1" : ""), text_factory(G.factory[CENTER, :]) )
    println(join(["-" for x in 1:50]))
    println("Bag : ",  join([string(G.bag[c]) * CHAR[c] * " " for c in 1:5]))
end