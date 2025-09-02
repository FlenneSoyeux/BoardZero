using Crayons

struct BoopMove <: AbstractMove
    c::Int64    #cell where it goes
    cat::Bool   #false for having a kitten
    promoted::Int   # The 3 cats in a row chosen (the alignement if > 0); or the cat being promoted (cell if < 0) ; or nothing if 0
    BoopMove(cell, cat) = new(cell, cat, 0)
    BoopMove(cell, cat, promoted) = new(cell, cat, promoted)
end

function Base.:(==)(m::BoopMove, n::BoopMove)    #necessary for comparing moves in keep_subtree! for instance
    return m.c == n.c && m.cat == n.cat && m.promoted == n.promoted
end

xy2cell(x, y) = x + 6*y - 6
cell2xy(cell) = mod(cell-1, 6)+1, div(cell-1, 6)+1

mutable struct Boop <: AbstractGame
    kittens::Vector{UInt64}
    cats::Vector{UInt64}
    playerToPlay::Int
    remaining_kittens::Vector{Int} # 1st player kittens 2ndplayer kittens
    remaining_cats::Vector{Int} # 1st player cats 2ndplayer cats
    finished::Bool
    ntour::Int          # at 100 rounds it's a draw (or 100 rounds without promotion ?)
    lastCell::Tuple{Int, Int}

    Boop() = new(UInt64[0,0], UInt64[0,0], 1, Int[8, 8], Int[0, 0], false, 0, (0,0))
end


const CHAR = [["$(Crayon(foreground=:blue)("k"))", "$(Crayon(foreground=:blue)("C"))"],
    ["$(Crayon(foreground=:yellow)("k"))", "$(Crayon(foreground=:yellow)("C"))"]]

function Game.is_finished(G::Boop) :: Bool
    return G.finished
end

function Game.get_winner(G::Boop) :: Int
    if G.ntour == 100
        return 0
    else
        return 3-G.playerToPlay
    end
end

function Game.number_played_moves(G::Boop) :: Int
    return G.ntour
end

function Game.get_printable_move(G::Boop, move::BoopMove) :: String
    x, y = cell2xy(move.c)
    return CHAR[G.playerToPlay][1+move.cat]*" "*string(x)*" "*string(y)*" "*(move.promoted != 0 ? "align : "*string(move.promoted) : "")
end
function Game.move_to_string(G::Boop, move::BoopMove) :: String
    return string(move.c)*" "*string(move.cat)*" "*string(move.promoted)
end
function Game.string_to_move(G::Boop, s::String)
    c, cat, promoted = split(s, ' ')
    c = parse(Int, c)
    cat = parse(Bool, cat)
    promoted = parse(Int, promoted)
    return BoopMove(c, cat, promoted)
end

function Game.get_null_move(G::Boop) :: BoopMove
    return BoopMove(-1, true)
end


function Base.print(G::Boop)
    println(G.lastCell, " lastcell")
    for x in 1:6, y in 1:6
        cell = cells[x+y*6-6]
        if G.kittens[1] & cell > 0
            print((x,y)==G.lastCell ? crayon"underline" : "", crayon"blue", " k ", crayon"reset")
        elseif G.cats[1] & cell > 0
            print((x,y)==G.lastCell ? crayon"underline" : "", crayon"blue", " C ", crayon"reset")
        elseif G.kittens[2] & cell > 0
            print((x,y)==G.lastCell ? crayon"underline" : "", crayon"yellow", " k ", crayon"reset")
        elseif G.cats[2] & cell > 0
            print((x,y)==G.lastCell ? crayon"underline" : "", crayon"yellow", " C ", crayon"reset")
        else
            print(" . ")
        end

        if y == 6
            if x == 2
                println(crayon"blue", "\tRemaining kittens : ", G.remaining_kittens[1], crayon"reset")
            elseif x== 3
                println(crayon"blue", "\tRemaining cats : ", G.remaining_cats[1], crayon"reset")
            elseif x == 5
                println(crayon"yellow", "\tRemaining kittens : ", G.remaining_kittens[2], crayon"reset")
            elseif x == 6
                println(crayon"yellow", "\tRemaining cats : ", G.remaining_cats[2], crayon"reset")
            else
                println()
            end
        end
    end
end


function Game.manual_input(G::Boop; newGame=true)
    if newGame
        return
    end
    for p in 1:2, cat in [false, true]
        print(G)
        doRestart = true
        while doRestart
            println("Print", cat ? "kittens" : "CATS", " of player ", p," Type s to stop, r to restart")
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
                    if cat
                        G.kittens[x, y, p] = 1
                    else
                        G.cats[x, y, p] = 1
                    end
                    
                end
            end
        end
    end

    println("Remaining kittens of 1 - cats of 1 - kittens of 2 - cats of 2")
    s = readline()
    G.remaining_kittens[1], G.remaining_cats[1], G.remaining_kittens[2], G.remaining_cats[2] = map(x->parse(Int,x), split(s, " "))

    print(G)
end