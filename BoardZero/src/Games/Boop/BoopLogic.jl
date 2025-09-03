
#FORMULA FOR CELL :
# CELL = X + 6*Y - 6
# X = mod(CELL-1,6)+1
# Y = div(CELL-1,6)+1

# const dcell = [-7, -6, -5, +1, +7, +6, +5, -1]
const dcell = [-7, -1, +5, +6, +7, +1, -5, -6]




function make_neighbors()
    #neighbor[cell, dir] gets the map to see if there is someone on this cell ; 0 if out
    neigh  = zeros(UInt64, 36, 8)
    neigh2 = zeros(UInt64, 36, 8)
    for c in 1:36
        for dir in 1:8
            if c % 6 == 1 && (12+dcell[dir])%6 == 5    #we hit top edge
                continue
            elseif c % 6 == 0 && (12+dcell[dir])%6 == 1    #we hit bottom edge
                continue
            elseif c+dcell[dir] > 36 || c+dcell[dir] <= 0 #we hit right or left edges
                continue
            end

            neigh[c, dir] = 1<<(c + dcell[dir]-1)

            if c % 6 == 2  && (12+dcell[dir])%6 == 5    #we hit left edges
                continue
            elseif c % 6 == 5 && (12+dcell[dir])%6 == 1    #we hit right edges
                continue
            elseif c+2*dcell[dir] > 36 || c+2*dcell[dir] <= 0 #we hit top or bot edges
                continue
            end
            neigh2[c, dir] = 1<<(c + 2*dcell[dir]-1)

        end
    end
    return neigh, neigh2
    #2nd neighbor[cell, dir] is the same for the 2nd cell
end

function make_alignements()
    A = UInt64[]
    for x in 1:6, y in 1:6
        c = x + y*6 - 6
        if x <= 4 && y <= 4   #diag
            push!(A, (1<<(c-1)) | (1<<(c-1+7)) | (1<<(c-1+14)))
        end
        if x >= 3 && y <= 4 #diag 2
            push!(A, (1<<(c-1)) | (1<<(c-1+5)) | (1<<(c-1+10)))
        end
        if x <= 4   #row
            push!(A, (1<<(c-1)) | (1<<(c-1+1)) | (1<<(c-1+2)))
        end
        if y <= 4   #col
            push!(A, (1<<(c-1)) | (1<<(c-1+6)) | (1<<(c-1+12)))
        end
    end
    return A
end

const cells = UInt64[1<<i for i in 0:35]
const neigh, neigh2 = make_neighbors()
const list_alignments = make_alignements()

function Game.play!(G::Boop, move::BoopMove)
    p = G.playerToPlay
    kittens = G.kittens[1] | G.kittens[2]
    board = kittens | G.cats[1] | G.cats[2]

    @assert board & cells[move.c] == 0

    #place
    if move.cat
        @assert G.remaining_cats[p] > 0
        G.cats[p] |= cells[move.c]
        G.remaining_cats[p] -= 1
        c = move.c

        #BOOP EVERYTHING
        for dir in 1:8
            if ((board & neigh[c, dir]) > 0) && ((neigh2[c, dir] & board) == 0)     # something boops !
                if G.kittens[1] & neigh[c, dir] > 0
                    if neigh2[c, dir] == 0   #kittens1 gets kicked out !
                        G.remaining_kittens[1] += 1
                        G.kittens[1] &= ~neigh[c, dir]
                    else
                        G.kittens[1] &= ~neigh[c, dir]
                        G.kittens[1] |= neigh2[c, dir]
                    end

                elseif G.kittens[2] & neigh[c, dir] > 0
                    if neigh2[c, dir] == 0   #kittens2 gets kicked out !
                        G.remaining_kittens[2] += 1
                        G.kittens[2] &= ~neigh[c, dir]
                    else
                        G.kittens[2] &= ~neigh[c, dir]
                        G.kittens[2] |= neigh2[c, dir]
                    end

                elseif G.cats[1] & neigh[c, dir] > 0
                    if neigh2[c, dir] == 0   #cats1 gets kicked out !
                        G.remaining_cats[1] += 1
                        G.cats[1] &= ~neigh[c, dir]
                    else
                        G.cats[1] &= ~neigh[c, dir]
                        G.cats[1] |= neigh2[c, dir]
                    end

                else
                    if neigh2[c, dir] == 0   #cats2 gets kicked out !
                        G.remaining_cats[2] += 1
                        G.cats[2] &= ~neigh[c, dir]
                    else
                        G.cats[2] &= ~neigh[c, dir]
                        G.cats[2] |= neigh2[c, dir]
                    end
                end
            end
        end

    else    # a kitten is placed
        @assert G.remaining_kittens[p] > 0
        G.kittens[p] |= cells[move.c]
        G.remaining_kittens[p] -= 1

        #BOOP EVERY KITTENS
        for dir in 1:8
            if ((kittens & neigh[move.c, dir]) > 0) && ((neigh2[move.c, dir] & board) == 0)     # something boops !
                if G.kittens[1] & neigh[move.c, dir] > 0
                    if neigh2[move.c, dir] == 0   #kittens1 gets kicked out !
                        G.remaining_kittens[1] += 1
                        G.kittens[1] &= ~neigh[move.c, dir]
                    else
                        G.kittens[1] &= ~neigh[move.c, dir]
                        G.kittens[1] |= neigh2[move.c, dir]
                    end

                else
                    if neigh2[move.c, dir] == 0   #kittens2 gets kicked out !
                        G.remaining_kittens[2] += 1
                        G.kittens[2] &= ~neigh[move.c, dir]
                    else
                        G.kittens[2] &= ~neigh[move.c, dir]
                        G.kittens[2] |= neigh2[move.c, dir]
                    end
                end
            end
        end
    end

    #now remove all the alignments
    if move.promoted == 0
        #no promotion


    elseif move.promoted < 0
        cell = cells[-move.promoted]
        if G.kittens[p] & cell > 0
            G.kittens[p] &= ~cell
            G.remaining_cats[p] += 1

        elseif G.cats[p] & cell > 0
            G.cats[p] &= ~cell
            G.remaining_cats[p] += 1
        else
            error("On devrait avoir un cat ou kitten ! ")
        end

    else
        mask = move.promoted
        @assert ((G.kittens[p] | G.cats[p]) & mask == mask)
        if G.kittens[p] & mask == 0
            #3 cats : it's a win !
            G.finished = true
        else
            G.remaining_cats[p] += 3
            G.kittens[p] &= ~mask
            G.cats[p] &= ~mask
        end

    end

    G.ntour += 1
    if G.ntour == 100
        G.finished = true
    end

    G.lastCell = cell2xy(move.c)
    G.playerToPlay = 3-G.playerToPlay
end

function list_align(board::UInt64)
    return UInt64[l for l in list_alignments if (board & l) == l]
end

function Game.all_moves(G::Boop) :: Vector{BoopMove}
    moves = Vector{BoopMove}()

    p = G.playerToPlay
    me = G.kittens[p]   | G.cats[p]
    he = G.kittens[3-p] | G.cats[3-p]
    kittens = G.kittens[1] | G.kittens[2]
    cats = G.cats[1] | G.cats[2]
    board = kittens | cats
    for (c,cell) in enumerate(cells)
        if cell & board > 0
            continue
        end

        # trying to put 1kit or 1cat in cell

        if G.remaining_kittens[p] > 0
            booped = UInt64(0)
            new_board = UInt64(0)
            cats_available = (G.remaining_cats[p] + G.remaining_kittens[p] > 1)   # a flag set to true if there will be an available cat or kitten to be played after ; otherwise player has the possibility to take 1 from game
            for dir in 1:8   #compute my booped kittens
                if ((me & kittens & neigh[c, dir]) > 0) && ((neigh2[c, dir] & board) == 0)   # to be booped by a kitten, neighbor kittens (not cats) must have nobody behind
                    booped |= neigh[c, dir]
                    if neigh2[c, dir] > 0   #it can move
                        new_board |= neigh2[c, dir]
                    else    #it goes out
                        cats_available = true
                    end
                end
            end

            new_board |= xor(me, booped) | cell
            #We got the new map !
            L = list_align(new_board)
            if(length(L) > 0)   #Some alignements occur !
                for l in L
                    push!(moves, BoopMove(c, false, l))
                end

            elseif !cats_available  # No alignement and all my cats are on baord !
                for (cc, check_cell) in enumerate(cells)
                    if check_cell & new_board > 0
                        push!(moves, BoopMove(c, false, -cc))
                    end
                end

            else    # Nothing happens
                push!(moves, BoopMove(c, false))

            end
        end

        if G.remaining_cats[p] > 0
            booped = UInt64(0)
            new_board = UInt64(0)
            cats_available = (G.remaining_cats[p] + G.remaining_kittens[p] > 1)
            for dir in 1:8
                if ((me & neigh[c, dir]) > 0) && ((neigh2[c, dir] & board) == 0)    # same but no kittens check : all my cats are boopable
                    booped |= neigh[c, dir]
                    if neigh2[c, dir] > 0   #it can move
                        new_board |= neigh2[c, dir]
                    else    #it goes out
                        cats_available = true
                    end
                end
            end

            new_board |= xor(me, booped) | cell #new board : previous board - booped + new position of booped + cell

            L = list_align(new_board)
            if length(L) > 0
                for l in L
                    push!(moves, BoopMove(c, true, l))
                end

            elseif !cats_available
                for (cc, check_cell) in enumerate(cells)
                    if check_cell & new_board > 0
                        push!(moves, BoopMove(c, true, -cc))
                    end
                end

            else
                push!(moves, BoopMove(c, true))

            end
        end

    end

    return moves
end



function Game.get_human_move(G::Boop)
    moves = all_moves(G)
    while true
        try
            println("Write k(or 'c') x y e.g. k 4 5 or c 1 1")
            cat, x, y = split(readline(), ' ')
            if cat != "k" && cat != "c"
                println("k or c")
                continue
            end
            x = parse(Int, x)
            y = parse(Int, y)
            if x < 1 || x > 6 || y < 1 || y > 6
                println("between 1 and 6")
                continue
            end
            s = sum([m.c == xy2cell(x, y) && m.cat == (cat=='c') for m in moves])
            if s == 0
                println("s = 0")
                display(moves)
                continue
            elseif s == 1
                for move in moves
                    if move.c == xy2cell(x, y) && move.cat == (cat=="c")
                        return move
                    end
                end
            else    # need to select good move
                choice = 0
                i = 1
                for move in moves
                    if m.c == move.c && m.cat == move.cat
                        println(i, " :\t", get_printable_move(G, move))
                        i += 1
                    end
                end
                while choice < 1 && choice > s
                    choice  = parse(Int, readline())
                end
                i = 1
                for move in moves
                    if m.c == move.c && m.cat == move.cat
                        if i == choice
                            return move
                        end
                        i += 1
                    end
                end
            end
        catch e
            println(e)
            if isa(e, InterruptException)
                rethrow(e)
            end
        end
    end
end


function random_move(G::Boop, rng)
    p = G.playerToPlay
    me = G.kittens[p]   | G.cats[p]
    he = G.kittens[3-p] | G.cats[3-p]
    kittens = G.kittens[1] | G.kittens[2]
    cats = G.cats[1] | G.cats[2]
    board = kittens | cats

    c = rand(rng, 1:36)
    while board & cells[c] > 0
        c = rand(rng, 1:36)
    end
    cell = cells[c]

    do_cat::Bool = true
    if G.remaining_cats[p] > 0 && G.remaining_kittens[p] > 0
        do_cat = rand(rng, 0:1)
    elseif G.remaining_cats[p] > 0
        do_cat = true
    else
        do_cat = false
    end

    if do_cat
        booped = UInt64(0)
        new_board = UInt64(0)
        cats_available = (G.remaining_cats[p] + G.remaining_kittens[p] > 1)
        for dir in 1:8
            if ((me & neigh[c, dir]) > 0) && ((neigh2[c, dir] & board) == 0)    # same but no kittens check : all my cats are boopable
                booped |= neigh[c, dir]
                if neigh2[c, dir] > 0   #it can move
                    new_board |= neigh2[c, dir]
                else    #it goes out
                    cats_available = true
                end
            end
        end

        new_board |= xor(me, booped) | cell #new board : previous board - booped + new position of booped + cell

        L = list_align(new_board)
        if length(L) > 0
            l = rand(rng, L)
            return BoopMove(c, true, l)

        elseif !cats_available
            cc = rand(rng, 1:36)
            while cells[cc] & new_board == 0
                cc = rand(rng, 1:36)
            end
            return BoopMove(c, true, -cc)

        else
            return BoopMove(c, true)

        end
    else
        booped = UInt64(0)
        new_board = UInt64(0)
        cats_available = (G.remaining_cats[p] + G.remaining_kittens[p] > 1)
        for dir in 1:8
            if ((me & kittens & neigh[c, dir]) > 0) && ((neigh2[c, dir] & board) == 0)    # same but no kittens check : all my cats are boopable
                booped |= neigh[c, dir]
                if neigh2[c, dir] > 0   #it can move
                    new_board |= neigh2[c, dir]
                else    #it goes out
                    cats_available = true
                end
            end
        end

        new_board |= xor(me, booped) | cell #new board : previous board - booped + new position of booped + cell

        L = list_align(new_board)
        if length(L) > 0
            l = rand(rng, L)
            return BoopMove(c, false, l)

        elseif !cats_available
            cc = rand(rng, 1:36)
            while cells[cc] & new_board == 0
                cc = rand(rng, 1:36)
            end
            return BoopMove(c, false, -cc)

        else
            return BoopMove(c, false)

        end
    end
end

function Game.play_random!(G::Boop, rng=default_rng())
    while !G.finished
        move = random_move(G, rng)
        Game.play!(G, move)
    end
end
