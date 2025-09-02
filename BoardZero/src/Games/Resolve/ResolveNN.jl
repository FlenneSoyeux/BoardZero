using Flux

#For NN
const NN_TYPE = "Conv"
const NUM_FILTERS = 64
const NUM_LAYERS = 4
const V_HEAD = 16
const PI_HEAD = 16
const SHAPE_INPUT = (5, 5, 4)
const SHAPE_OUTPUT = (5, 5, 3)
const NUMBER_OUTPUTS = 76
const NUMBER_ACTIONS = 75     #Total number of actions (=size of pi_output)
const LENGTH_POINTS = 0

# 4 FILTERS :
# playerToPlay board
# other player board
# resolves up only
# resolves left only

#OUTPUT : 3 filters
# probability for placing in (x,y)
# probability for resolving up from (x,y) Warning : in NS framework !!
# probability for resolving left from (x,y)  Warning : in NS framework !!

function Game.load_from(Gfrom::Resolve, Gto::Resolve)
    Gto.board .= Gfrom.board
    Gto.playerToPlay = Gfrom.playerToPlay
    Gto.lastMove = Gfrom.lastMove
    Gto.lastLastCell = Gfrom.lastLastCell
    Gto.isFinished = Gfrom.isFinished
    Gto.winner = Gfrom.winner
    Gto.turnSamePlayer = Gfrom.turnSamePlayer
    Gto.remainingCells = Gfrom.remainingCells
end


function Game.get_input_for_nn(G::Resolve)
    input_pos::Array{Float32, 3} = cat(reshape(G.board[:, G.playerToPlay], 5, 5), reshape(G.board[:, 3-G.playerToPlay], 5, 5), zeros(Float32, 5, 5), zeros(Float32, 5, 5); dims=3)

    if G.playerToPlay == 2  # need to transform EW to NS !
        input_pos[:, :, 1] = rotl90(input_pos[:, :, 1])
        input_pos[:, :, 2] = rotl90(input_pos[:, :, 2])
        # cell previously on (x, y) is now on (6-y, x)
    end

    if (G.lastMove != 0 && G.board[G.lastMove, G.playerToPlay]) || G.remainingCells == 0   # resolve! 
        for move in all_moves(G)
            x, y = cell2xy(move.cell)
            if move.kind == 1   # xy Up
                if G.playerToPlay == 1
                    input_pos[x, y, 3] = 1.0
                else    # xy - x-1y so now : (6-y,x)-(6-y,x-1) it's a (6-y,x)left
                    input_pos[6-y, x, 4] = 1.0
                end
            elseif move.kind == 2   # xy Left
                if G.playerToPlay == 1
                    input_pos[x, y, 4] = 1.0
                else    # xy - xy-1 so now : (6-y,x)-(7-y,x) it's a (7-y,x)up
                    input_pos[7-y, x, 3] = 1.0
                end
            elseif move.kind == 3   # xy Down : x+1 y up
                if G.playerToPlay == 1
                    input_pos[x+1, y, 3] = 1.0
                else    # xy - x+1y so now : (6-y,x)-(6-y,x+1) it's a (6-y,x+1)left
                    input_pos[6-y, x+1, 4] = 1.0
                end
            elseif move.kind == 4   # Right : x y+1 left
                if G.playerToPlay == 1
                    input_pos[x, y+1, 4] = 1.0
                else    # xy - xy+1 so now : (6-y,x)-(5-y,x) it's a (6-y,x)up
                    input_pos[6-y, x, 3] = 1.0
                end
            else
                error("move.kind ", move.kind)
            end
        end
        
    end

    return input_pos
end


function Game.get_input_for_nn!(G::Resolve, pi_MCTS::Vector{Float32}, rng)
    input = get_input_for_nn(G)
    return input
    
    pi_MCTS = reshape(pi_MCTS, 5, 5, 3)

    if rand(rng, Bool)  # axial symmetry
        input[:, :, 1:2] = reverse(input[:, :, 1:2]; dims=2)
        pi_MCTS .= reverse(pi_MCTS; dims=2)
    end

    r = rand(rng, 1:4)  # rotation
    if r == 1
        for i in 1:3
            input[:, :, i] = rotr90(input[:, :, i])
        end
        pi_MCTS[:, :] .= rotr90(pi_MCTS[:, :])
        input[:, :, 4] = 1 .- input[:, :, 4]    #NS becomes SE and conversely

    elseif r == 2
        for i in 1:3
            input[:, :, i] = rot180(input[:, :, i])
        end
        pi_MCTS[:, :] .= rot180(pi_MCTS[:, :])

    elseif r == 3
        for i in 1:3
            input[:, :, i] = rotl90(input[:, :, i])
        end
        pi_MCTS[:, :] .= rotl90(pi_MCTS[:, :])
        input[:, :, 4] = 1 .- input[:, :, 4]    #NS becomes SE and conversely
    end

    pi_MCTS = reshape(pi_MCTS, :)
    return input
end


function Game.get_idx_output(G::Resolve, move::ResolveMove) :: Int
    #output : place filter ; resolve UP filter ; resolve LEFT filter
    #warning : in NS framework. So if playerToPlay is 2, (x,y) cell in board is actually in (6-y, x) in filters
    if G.playerToPlay == 1
        if move.kind == 0
            return move.cell
        elseif move.kind == 1   # up
            return 25 + move.cell
        elseif move.kind == 2 # left
            return 50 + move.cell
        elseif move.kind == 3   # down : so take (x+1,y) instead
            return 25 + move.cell+1
        elseif move.kind == 4   #right : so take (x, y+1) instead
            return 50 + move.cell+5
        end
    else
        x, y = cell2xy(move.cell)
        cell = xy2cell(6-y, x)
        if move.kind == 0
            return cell
        elseif move.kind == 1   # up therefore after rotl90 : left
            return 50 + cell
        elseif move.kind == 2 # left therefore after rotl90 : down
            return 25 + cell+1
        elseif move.kind == 3   # down therefore after rotl90 : right
            return 50 + cell+5
        elseif move.kind == 4   # right : therefore after rotl90 : up
            return 25 + cell
        end
    end
end

# After output of nn, we chose move number idx. What move is it corresponding to ?
function Game.get_move_from_idx_move(G::Resolve, idx::Int) :: ResolveMove
    #warning : in NS framework. 
    # So if playerToPlay is 2, (x,y) cell in board is actually in (6-y, x) in filters
    # Meaning that (x, y) in filters is (y, 6-x)
    if 1 <= idx <= 25
        if G.playerToPlay == 1
            return ResolveMove(idx, 0)
        else
            x, y = cell2xy(idx)
            return ResolveMove( xy2cell(y, 6-x), 0 )
        end

    elseif 26 <= idx <= 50  # up
        if G.playerToPlay == 1
            return ResolveMove(idx-25, 1)
        else
            x, y = cell2xy(idx-25)
            @assert 6-x < 5
            return ResolveMove( xy2cell(y, 6-x), 4 ) # right
        end
    else    # left
        if G.playerToPlay == 1
            return ResolveMove(idx-50, 2)
        else
            x, y = cell2xy(idx-50)
            @assert y > 1
            return ResolveMove( xy2cell(y, 6-x), 1 ) # up
        end
    end
end


function Game.print_input_nn(::Resolve, input, pi_MCTS; print_pos=true)
    G = new_game()
    G.board[:, 1] = reshape(input[:, :, 1], 25)
    G.board[:, 2] = reshape(input[:, :, 2], 25)

    if print_pos
        print(G)
    end

    v = [(i, x) for (i, x) in enumerate(pi_MCTS)]
    sort!(v, by=x->-x[2])
    for (i,p) in v[1:min(length(v),5)]
        if p == 0
            break
        end
        move = get_move_from_idx_move(G, i)
        println(round(Int, 100*p), " %\t", get_printable_move(G, move))
    end
end




## Model 
struct ResolveNN <: AbstractNN
    trunk::Chain
    pi_head::Chain
    v_head::Chain
    name::String
    ResolveNN() = new(
        Chain(Conv((3,3), SHAPE_INPUT[3]=>NUM_FILTERS, pad=1, bias=false), BatchNorm(NUM_FILTERS, relu), 
                [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:(NUM_LAYERS - 2 - 2*div(NUM_LAYERS-2,3))]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...) |> gpu,
        Chain(Conv((3,3), NUM_FILTERS=>PI_HEAD, pad=1, bias=false), BatchNorm(PI_HEAD, relu), Conv((3,3), PI_HEAD=>SHAPE_OUTPUT[3], pad=1), Flux.flatten) |> gpu, # pi
        Chain(Conv((3,3), NUM_FILTERS=>V_HEAD, pad=1, bias=false), BatchNorm(V_HEAD, relu), Flux.flatten, Dense(V_HEAD*25 => 1, sigmoid)) |> gpu, # v
        "b"*string(NUM_LAYERS)*"c"*string(NUM_FILTERS)*"_SE"
    )
    ResolveNN(trunk, pi_head, v_head, name) = new(trunk, pi_head, v_head, name)
end

function NNet.predict(model::ResolveNN, input)
    @assert ndims(input) == 4
    y = model.trunk(input)
    return model.pi_head(y), model.v_head(y)
end

function NNet.predict(model::ResolveNN, G::Resolve)
    input = add_last_dim(model, get_input_for_nn(G)) |> gpu
    output = predict(model, input) |> cpu
    piNN, vNN = squeeze(model, output)
    return piNN, vNN[1]
end

function NNet.get_input_shape(::ResolveNN, batchsize::Int)
    return zeros(Float32, (SHAPE_INPUT..., batchsize))
end
function NNet.get_output_shape(::ResolveNN, batchsize::Int)
    return (zeros(Float32, NUMBER_ACTIONS, batchsize), zeros(Float32, 1, batchsize)) # pi, v
end

function NNet.concatenate_inputs(::ResolveNN, inputs)
    return [     [ input[x, y, z] for x in 1:SHAPE_INPUT[1], y in 1:SHAPE_INPUT[2], z in 1:SHAPE_INPUT[3], input in inputs ]      ]
end

function NNet.inplace_change!(::ResolveNN, inputs, i, input)
    # inputs[i] <- input it's made s.t. model.predict(inputs) is directly callable
    inputs[:, :, :, i] = input
end

function NNet.add_last_dim(::ResolveNN, input) # go to 3dim to 4 dim (for convolutional networks)
    return reshape(input, size(input)..., 1)
end
function NNet.squeeze(::ResolveNN, output)
    return dropdims(output[1]; dims=2), dropdims(output[2]; dims=2)
end


function NNet.loss(model::ResolveNN, input, target)
    pi_nn, v_nn = predict(model, input)
    Lv = Flux.mse(v_nn[1, :], target[NUMBER_ACTIONS+1, :])
    Lpi = Flux.Losses.logitcrossentropy(pi_nn, target[1:NUMBER_ACTIONS, :]; dims=1)   +    sum(  Flux.Losses.xlogx.(target[1:NUMBER_ACTIONS, :]) ) / size(target)[2]       #Kullback Leibler
    return Lv+Lpi, Lpi, Lv
end

function NNet.load(::Resolve, id)
    BSON.@load DIRECTORY*"model_"*string(id)*".bson" trunk pi_head v_head name
    nn = ResolveNN(trunk |> gpu, pi_head |> gpu, v_head |> gpu, name)
    println("Model ", id, " loaded")
    return nn
end
function NNet.save(nn::ResolveNN, id_model)
    trunk = nn.trunk |> cpu
    pi_head= nn.pi_head |> cpu
    v_head = nn.v_head |> cpu
    name = nn.name
    BSON.@save DIRECTORY*"model_"*string(id_model)*".bson" trunk pi_head v_head name
end
