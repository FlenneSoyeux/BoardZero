
const dpos_char = ['↑','←','↓','→']
const MAX_TURN_SAME_PLAYER = 6

struct ResolveMove <: AbstractMove
    cell::Int
    kind::Int   #0 : place ; 1 : up ; 2 : left ; 3 : down ; 4 : right
        # Only up and left in second phase
end


cell2xy(cell) = mod(cell-1, 5)+1, div(cell-1, 5)+1
xy2cell(x, y) = x+y*5-5

# For each cell, tells if direction dir (from 1 to 4) is possible or not
function make_dir_possible()    
    arr = zeros(Bool, SIZE*SIZE, 4)
    for x in 1:SIZE, y in 1:SIZE
        cell = xy2cell(x, y)
        arr[cell, 1] = (x>1)    # Up
        arr[cell, 2] = (y>1)    # Left
        arr[cell, 3] = (x<SIZE) # Down
        arr[cell, 4] = (y<SIZE) # Right
    end
    return arr
end

const SIZE = 5
const DIR = [-1, -SIZE, 1, SIZE]    # up left down right
const DIR_POSSIBLE = make_dir_possible()    # DIR_POSSIBLE[cell, dir] is true/false : depending on if the move is possible


function Base.:(==)(m::ResolveMove, n::ResolveMove)    #necessary for comparing moves in keep_subtree! for instance
    return m.cell == n.cell && m.kind == n.kind
end


mutable struct Resolve <: AbstractGame
    board::Array{Bool, 2}   # Player 1, player 2
    playerToPlay::Int
    lastMove::Int
    lastLastCell::Int   # can't do A -> B -> A swaps
    isFinished::Bool
    winner::Int
    turnSamePlayer::Int # Limit to 10 to avoid infinite loops
    remainingCells::Int # If 0 : only resolves

    Resolve() = new( zeros(Bool, 25, 2) , 1, 0, 0, false, -1, MAX_TURN_SAME_PLAYER, SIZE*SIZE)
end



function Game.is_finished(G::Resolve) :: Bool
    return G.isFinished
end

function Game.get_winner(G::Resolve) :: Int
    @assert G.winner != -1
    return G.winner
end

function Game.number_played_moves(G::Resolve) :: Int
    return sum(G.board)
end


function Game.get_printable_move(G::Resolve, move::ResolveMove) :: String
    if move.kind == 0
        x, y = cell2xy(move.cell)
        return string(x)*" "*string(y)*" PLACE"
    else
        x, y = cell2xy(move.cell)
        return string(x)*" "*string(y)*" swap"*dpos_char[move.kind]
    end
end

function Game.move_to_string(G::Resolve, move::ResolveMove) :: String
    return string(move.cell)*" "*string(move.kind)
end

function Game.string_to_move(G::Resolve, s::String)
    c, k, promoted = split(s, ' ')
    c = parse(Int, c)
    k = parse(Bool, k)
    return ResolveMove(c, k)
end
function Game.get_null_move(G::Resolve) :: ResolveMove
    return ResolveMove(-1, -1)
end

const CHAR = ["$(Crayon(foreground=:blue)("x"))", "$(Crayon(foreground=:red)("o"))"]

function Base.print(G::Resolve)
    println("last move : ", G.lastMove)
    print("  ")
    for y in 1:5
        print(" ", crayon"blue", y, " ", crayon"reset")
    end
    println()
    for x in 1:5
        print(crayon"red", x, " ", crayon"reset")
        for y in 1:5
            cell = xy2cell(x, y)
            if G.board[cell, 1]
                print(" ", cell == G.lastMove ? crayon"underline" : "", CHAR[1], crayon"reset", " ")
            elseif G.board[cell, 2]
                print(" ", cell == G.lastMove ? crayon"underline" : "", CHAR[2], crayon"reset", " ")
            else
                print(" . ")
            end
        end
        println(crayon"red", "|", crayon"reset")
    end
    println(crayon"blue", "   ---------------", crayon"reset")
    println(CHAR[G.playerToPlay], " to play")
end



function Game.manual_input(G::Resolve; newGame=true)
    if newGame
        return
    end
    for p in 1:2
        print(G)
        doRestart = true
        while doRestart
            println("Print positions of player ", (p==1) ? "N-S" : "E-W"," Type s to stop, r to restart")
            doRestart = false
            while true
                input = readline()
                if input == "s"
                    break
                elseif input == "r"
                    doRestart = true
                    G.board[:, p] *= 0
                    break
                else
                    x, y = split(input, " ")
                    x, y = parse(Int, x), parse(Int, y)
                    G.board[xy2cell(x, y), p] = 1
                end
            end
        end
    end
    G.playerToPlay = (sum(G.board[:, 1]) == sum(G.board[:, 2])) ? 1 : 2

    print(G)
end