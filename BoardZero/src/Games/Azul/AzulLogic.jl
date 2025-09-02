
function Game.play!(G::Azul, move::AzulMove)
    if move.kind == ACTION_RANDOM   # Fill every factories back with tiles from bag
        @assert move.seed != 0 && G.playerToPlay == 0
        Random.seed!(G.core_rng, move.seed)

        G.round += 1
        G.playerToPlay = findmax(G.hasFirstPlayerTile)[2]
        G.hasFirstPlayerTile *= 0
        for line in 1:FACTORIES, qty in 1:4
            if sum(G.bag) == 0 #need to refill !
                G.bag = [20, 20, 20, 20, 20]
                for line2 in 1:5, color in 1:5
                    G.bag[color] -= G.factory[line2, color]
                end
                for p in 1:NUM_PLAYERS, x in 1:5, y in 1:5
                    G.bag[ BOARDCOLOR[x, y] ] -= G.board[x, y, p]
                end
                for p in 1:NUM_PLAYERS, x in 1:5
                    if G.pattern[p, x][1] > 0
                        G.bag[G.pattern[p, x][2]] -= G.pattern[p, x][1]
                    end
                end
            end
            color = draw(G.core_rng, G.bag)
            G.bag[color] -= 1
            G.factory[line, color] += 1
        end


    elseif move.kind == ACTION_TAKE
        if move.line == CENTER #take all of that color, and first player tile
            G.buffer[move.color] = G.factory[CENTER, move.color]
            G.factory[CENTER, move.color] = 0
            if !G.hasFirstPlayerTile[1] && !G.hasFirstPlayerTile[2]
                G.hasFirstPlayerTile[G.playerToPlay] = true
                G.floor[G.playerToPlay] += 1
            end

        else    # take all of that color from the factory, and the remaining go to center
            G.buffer[move.color] = G.factory[move.line, move.color]
            G.factory[move.line, move.color] = 0
            for color in 1:5
                G.factory[CENTER, color] += G.factory[move.line, color]
                G.factory[move.line, color] = 0
            end
        end

    else   #ACTION_PLACE
        if move.line == FLOOR_LINE
            G.floor[G.playerToPlay] += G.buffer[move.color]
            G.buffer[move.color] = 0

        else    # simply place on pattern line
            qty, color = G.pattern[G.playerToPlay, move.line]
            @assert (qty == 0 || color == move.color) && qty < move.line

            diff = min(G.buffer[move.color], move.line-qty)
            G.buffer[move.color] -= diff
            qty += diff
            G.floor[G.playerToPlay] += G.buffer[move.color]
            G.buffer[move.color] = 0

            G.pattern[G.playerToPlay, move.line] = (qty, move.color)

        end

        G.playerToPlay = (G.playerToPlay == NUM_PLAYERS) ? 1 : G.playerToPlay+1

        if sum(G.factory) == 0  # ROUND FINISHED
            #points
            for line in 1:5, p in 1:NUM_PLAYERS
                if G.pattern[p, line][1] == line    # pattern line is full
                    color = G.pattern[p, line][2]
                    G.scores[p] += place_tile(G.board[:, :, p], line, color)
                    G.board[line, COLORPOS[line, color], p] = true
                    G.pattern[p, line] = (0, 0)
                end
            end

            #floor penalties
            for p in 1:NUM_PLAYERS
                G.scores[p] -= min(G.scores[p], floor_points(G.floor[p]))
                G.floor[p] = 0
            end

            #Game is finished ?
            G.finished = any(sum(G.board; dims=2) .== 5) || G.round == 8
            if G.finished
                #end game
                for p in 1:NUM_PLAYERS
                    G.scores[p] += 2*count(sum(G.board[:, :, p]; dims=2) .== 5) #every line
                    G.scores[p] += 7*count(sum(G.board[:, :, p]; dims=1) .== 5) #every column
                    for color in 1:5
                        if G.board[1, COLORPOS[1, color], p] && G.board[2, COLORPOS[2, color], p] && G.board[3, COLORPOS[3, color], p] && G.board[4, COLORPOS[4, color], p] && G.board[5, COLORPOS[5, color], p]
                            G.scores[p] += 10
                        end
                    end
                end
            end

            G.playerToPlay = 0  # to tell there is a new round and random stuffs to do
        end
    end
end

function all_moves(G::Azul) :: Vector{AzulMove}
    @assert G.playerToPlay != 0 "cant all_moves if computer is playing"
    moves = []
    if sum(G.buffer) == 0 #do ACTION_TAKE
        for factory in 1:CENTER
            for color in 1:5
                if G.factory[factory, color] > 0
                    push!(moves, AzulMove(ACTION_TAKE, factory, color, 0))
                end
            end
        end

    else    #do ACTION_PLACE
        color = 1
        while G.buffer[color] == 0
            color += 1
        end

        for line in 1:5
            qty, color_pattern = G.pattern[G.playerToPlay, line]
            if (qty == 0 && !G.board[line, COLORPOS[line, color], G.playerToPlay]) || (qty > 0 && color == color_pattern && qty < line)
                push!(moves, AzulMove(ACTION_PLACE, line, color, 0))
            end
        end

        push!(moves, AzulMove(ACTION_PLACE, FLOOR_LINE, color, 0))

    end
    return moves
end

function Game.manual_input(G::Azul; newGame=false) 
    # input that crushes previous data

    println("Player 1 is the player to play")
    G.playerToPlay = 1
    G.finished = false
    G.round = 1
    G.hasFirstPlayerTile = [false, false]
    G.board .= zeros(Bool, 5, 5, 2)
    G.buffer .= [0,0,0,0,0]
    G.bag .= [20, 20, 20, 20, 20]
    G.scores .= [0,0]
    G.floor .= [0,0]
    for facto in 1:CENTER
        G.factory[facto, :] .= [0, 0, 0, 0, 0]
    end
    for line in 1:5
        G.pattern[1, line] = (0,0)
        G.pattern[2, line] = (0,0)
    end

    if !newGame while true
        println("Enter score :")
        str = readline()
        try 
            a = [parse(Int, x) for x in split(str, ' ')]
            G.scores .= a
        catch
            continue
        end
        break
    end end

    if !newGame while true
        println("Who has first player tile ? 1, 2 or nobody (0)")
        str = readline()
        try 
            a = parse(Int, str)
            if a == 1
                G.hasFirstPlayerTile[1] = true
            elseif a == 2
                G.hasFirstPlayerTile[2] = true
            end
        catch
            continue
        end
        break
    end end

    if !newGame while true
        println("Enter floors (including first player tile) :")
        str = readline()
        try 
            a = [parse(Int, x) for x in split(str, ' ')]
            G.floor .= a
            # TODO : check if G.floor >= G.hasFirstPlayerTile
        catch
            continue
        end
        break
    end end

    print(G)

    if !newGame for p in 1:2
         while true
            println("Enter data for player ", p)
            println("write 5 lines like 01101")
            for line in 1:5
                while true
                    print("line ", line, " : ")
                    str = readline()
                    if length(str) != 5 || any([!(s in ['0', '1']) for s in str])
                        continue
                    end
                    for y in 1:5
                        G.board[line, y, p] = (str[y] == '1')
                    end
                    break
                end
            end
            
            restart = true
            while true
                print(G)
                println("Add pattern line ? '3 AA' for AA in line 3\t'y' to accept 'r' to restart player")
                str = readline()
                if str == "y"
                    restart = false
                    break
                elseif str == "r"
                    restart = true
                    break
                end
                x, tiles = split(str, ' ')
                x = parse(Int64, x)
                color = Dict(['A'=>1, 'Y'=>2, 'R'=>3, 'B'=>4, 'W'=>5])[tiles[1]]
                G.pattern[p, x] = (length(tiles), color)
            end

            if restart
                println("Cancelling...")
                G.board[:, :, p] .= zeros(Bool, 5, 5)
                for line in 1:5
                    G.pattern[p, line] = (0,0)
                end
                continue
            else
                break
            end
        end
    end end

    while true
        try
            for facto in 1:(newGame ? CENTER-1 : CENTER)
                (facto != CENTER) ? println("Print factory ", facto, ". E for empty") : println("Print center. Do not count first tile. E for empty")
                str = readline()
                if str == "E"
                    continue
                else
                    for s in str
                        G.factory[facto, Dict(['A'=>1, 'Y'=>2, 'R'=>3, 'B'=>4, 'W'=>5])[s]] += 1
                    end
                end
            end
            print(G)
            println("Accept factories ? 'y'")
            str = readline()
            if str == "y" || str == "Y"
                break
            else
                println("Cancelling...")
                for facto in 1:CENTER
                    G.factory[facto, :] .= [0,0,0,0,0]
                end

                continue
            end
        catch e
            println(e)
            for facto in 1:CENTER
                G.factory[facto, :] .= [0,0,0,0,0]
            end
            println("Error - redo factories")
        end
    end

    for line in 1:5, p in 1:2
        qty, color = G.pattern[p, line]
        if qty != 0
            G.bag[ color ] -= qty
        end
        for y in 1:5
            if G.board[line, y, p] 
                G.bag[ BOARDCOLOR[line, y] ] -= line
            end 
        end
    end
    for facto in 1:CENTER, color in 1:5
        G.bag[color] -= G.factory[facto, color]
    end
    println("sum : ", sum(G.bag))
    if !newGame while true
        println("Automatically compute bag ? currently : ", G.bag, " y/n")
        str = readline()
        if str == "y"
            while sum(G.bag)%20 >= 5
                for c in 1:5
                    G.bag[c] = max(0, G.bag[c]-1)
                end
            end
            while sum(G.bag) % 20 > 0
                c = rand(1:5)
                if G.bag[c] > 0
                    G.bag[c] -= 1
                end
            end

            break
        elseif str == "n"
            println("Enter bag composition : ")
            G.bag = parse.(Int, split(readline(), ' '))
            println("Accept ", G.bag, "?")
            if readline() == "y"
                break
            end
        else
            println("retype y/n")
        end
    end end

    print(G)
end

function Game.get_human_move(G::Azul)
    moves = all_moves(G)

    if moves[1].kind == ACTION_TAKE
        for (i, m) in enumerate(moves)
            println(i, ":", get_printable_move(G, m))
        end
        i = 0
        while i < 1 || i > length(moves)
            println("Type move :")
            try
                i = parse(Int64, readline())
            catch e
                if e isa InterruptException
                    rethrow(e)
                end
                i = 0
            end
        end
        return moves[i]

    elseif moves[1].kind == ACTION_PLACE
        for (i, m) in enumerate(moves)
            println(m.line, ":", get_printable_move(G, m))
        end
        lines = [m.line for m in moves]
        i = 0
        while !(i in lines)
            println("Type move :")
            try
                i = parse(Int64, readline())
            catch e
                if e isa InterruptException
                    rethrow(e)
                end
                i = 0
            end
        end
        return moves[findall( m->m.line == i, moves )[1]]


    end

    for (i, m) in enumerate(moves)
        println(i, ":", get_printable_move(G, m))
    end
    i = 0
    while i < 1 || i > length(moves)
        println("Type move :")
        try
            i = parse(Int64, readline())
        catch e
            if e isa InterruptException
                rethrow(e)
            end
            i = 0
        end
    end

    return moves[i]
end


function random_move(G::Azul, rng) :: AzulMove
    return G.playerToPlay == 0 ? get_move_random_stuffs(G, rng)  : rand(rng, all_moves(G))
end

function Game.play_random!(G::Azul, rng=default_rng())
    while !G.finished
        move = random_move(G, rng)
        Game.play!(G, move)
    end
end

function Game.get_move_random_stuffs(G::Azul, rng=Random.default_rng())
    return AzulMove(ACTION_RANDOM, 0, 0, rand(rng, Int))
end

function Game.get_human_random_stuffs(G::Azul)
    G.round += 1
    G.playerToPlay = findmax(G.hasFirstPlayerTile)[2]
    G.hasFirstPlayerTile *= 0

    while true
        for facto in 1:FACTORIES
            println("Print factory ", facto, ". E for empty")
            str = readline()
            if str == "E"
                continue
            else
                for s in str
                    G.factory[facto, Dict(['A'=>1, 'Y'=>2, 'R'=>3, 'B'=>4, 'W'=>5])[s]] += 1
                end
            end
        end
        print(G)
        println("Accept factories ? 'y'")
        str = readline()
        if str == "y" || str == "Y"
            break
        else
            println("Cancelling...")
            for facto in 1:CENTER
                G.factory[facto, :] .= [0,0,0,0,0]
            end
            continue
        end
    end

    for facto in 1:FACTORIES, color in 1:5
        G.bag[color] -= G.factory[facto, color]
    end
    println("Du coup bag = ", G.bag)

end