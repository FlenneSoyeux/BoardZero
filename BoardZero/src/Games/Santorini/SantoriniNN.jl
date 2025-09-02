using Flux

const NN_TYPE = "Conv"
const NUM_FILTERS = 128
const NUM_LAYERS = 10
const V_HEAD = 64
const PI_HEAD = 64
const SHAPE_INPUT = (5, 5, 12)
const SHAPE_OUTPUT = (5, 5, 6)
const NUMBER_ACTIONS = 150     #Total number of actions (=size of pi_output)
const NUMBER_OUTPUTS = 151
const LENGTH_POINTS = 0


# INPUT : 
# filter level 0
# filter level 1
# filter level 2
# filter level 3
# filter level 4
# filter worker 1
# filter worker 2
# filter opponent workers
# filter worker 1 can move to
# filter worker 2 can move to
# filter worker 1 can build to (1 or 2 or to4)
# filter worker 2 can build to (1 or 2 or to4)
# parameters : gods

# OUTPUT :
# filter move worker 1
# filter move worker 2
# filter build worker 1
# filter build worker 2
# filter special worker 1
# filter special worker 2


#AZ + model STUFF
function Game.load_from(Gfrom::Santorini, Gto::Santorini)
    Gto.board .= Gfrom.board
    Gto.workers .= Gfrom.workers  # workers 1 2 3 4 (1 and 2 for player 1, 3 and 4 for player 2)
    Gto.gods .= Gfrom.gods
    Gto.buildWorker = Gfrom.buildWorker
    Gto.winner = Gfrom.winner
    Gto.playerToPlay = Gfrom.playerToPlay
    Gto.isFinished = Gfrom.isFinished
    Gto.godFlag .= Gfrom.godFlag
end

# Sometimes we can't perform output_nn[move] directly. We need to find idx to do output_nn[idx] after
function Game.get_idx_output(G::Santorini, move::SantoriniMove) :: Int
    if move.kind == ACTION_MOVE
        return 25*move.idWorker-25 + move.cell
    elseif move.kind == ACTION_BUILD
        return 25*move.idWorker+25 + move.cell
    else
        return 25*move.idWorker+75 + move.cell
    end
end

# After output of nn, we chose move number idx. What move is it corresponding to ?
function Game.get_move_from_idx_move(G::Santorini, idx::Int) :: SantoriniMove
    filter, cell = div(idx-1, 25)+1, mod(idx-1, 25)+1
    id = mod(filter-1, 2)+1
    if 1 <= filter <= 2
        return SantoriniMove(id, cell, ACTION_MOVE)
    elseif 3 <= filter <= 4
        return SantoriniMove(id, cell, ACTION_BUILD)
    else
        return SantoriniMove(id, cell, ACTION_SPECIAL)
    end

end

function Game.get_input_for_nn(G::Santorini)
    # input : levels 1, 2, 3, 4, where id1 is, where id2 is, where opponents are
    input = zeros(Float32, 5, 5, 12)
    board = reshape(G.board, 5, 5)
    input[:, :, 1] .= board.==0
    input[:, :, 2] .= board.==1
    input[:, :, 3] .= board.==2
    input[:, :, 4] .= board.==3
    input[:, :, 5] .= board.==4

    input[cell2xy(G.workers[G.playerToPlay, 1])..., 6] = 1.0f0
    input[cell2xy(G.workers[G.playerToPlay, 2])..., 7] = 1.0f0

    input[cell2xy(G.workers[3-G.playerToPlay, 1])..., 8] = 1.0f0
    input[cell2xy(G.workers[3-G.playerToPlay, 2])..., 8] = 1.0f0

    # can move to and can build to filters
    moves = all_moves(G)
    for move in moves
        if move.kind == ACTION_MOVE
            input[cell2xy(move.cell)..., 8+move.idWorker] = 1.0
        elseif move.kind == ACTION_BUILD
            input[cell2xy(move.cell)..., 10+move.idWorker] = 1.0
        else
            # pass for ATLAS
        end
    end

    parameters = zeros(Float32, 22)
    parameters[G.gods[G.playerToPlay]]   = 1
    parameters[11+G.gods[3-G.playerToPlay]] = 1
    

    return [input, parameters]
end


function Game.get_input_for_nn!(G::Santorini, pi_MCTS::Vector{Float32}, rng = Random.default_rng())
    input, parameters = get_input_for_nn(G)
    pi_MCTS = reshape(pi_MCTS, SHAPE_OUTPUT)
    #apply axial symmetry
    if rand(rng, Bool)
        input = reverse(input; dims = 2)
        pi_MCTS .= reverse(pi_MCTS; dims=2)
    end

    #apply change of 1-2
    if rand(rng, Bool)
        input[:, :, 6], input[:, :, 7] = input[:, :, 7], input[:, :, 6]
        input[:, :, 9], input[:, :, 10] = input[:, :, 10], input[:, :, 9]
        input[:, :, 11], input[:, :, 12] = input[:, :, 12], input[:, :, 11]
        pi_MCTS[:, :, 1], pi_MCTS[:, :, 2] = pi_MCTS[:, :, 2], pi_MCTS[:, :, 1]
        pi_MCTS[:, :, 3], pi_MCTS[:, :, 4] = pi_MCTS[:, :, 4], pi_MCTS[:, :, 3]
        pi_MCTS[:, :, 5], pi_MCTS[:, :, 6] = pi_MCTS[:, :, 6], pi_MCTS[:, :, 5]
    end

    #apply rotational symmetry
    r = rand(rng, 0:3)
    if r == 1
        for i in 1:12
            input[:,:,i] = rotr90(input[:,:,i])
        end
        for i in 1:6
            pi_MCTS[:, :, i] = rotr90(pi_MCTS[:, :, i])
        end

    elseif r == 2
        for i in 1:12
            input[:,:,i] = rot180(input[:,:,i])
        end
        for i in 1:6
            pi_MCTS[:, :, i] = rot180(pi_MCTS[:, :, i])
        end

    elseif r == 3
        for i in 1:12
            input[:,:,i] = rotl90(input[:,:,i])
        end
        for i in 1:6
            pi_MCTS[:, :, i] = rotl90(pi_MCTS[:, :, i])
        end

    end

    pi_MCTS = reshape(pi_MCTS, NUMBER_ACTIONS)

    return [input, parameters]
end


function Game.print_input_nn(::Santorini, inputs, pi_MCTS ; print_pos=true)
    # where 1 moves - where 1 builds - where 2 moves - where 2 builds
    input, params = inputs
    G = new_game()
    G.gods[1] = [i for (i,x) in enumerate(params[1:11]) if x > 0][1]
    G.gods[2] = [i for (i,x) in enumerate(params[12:22]) if x > 0][1]
    G.playerToPlay = 1
    idWorker2 = 1
    for x in 1:5, y in 1:5
        G.board[xy2cell(x, y)] = 1*(input[x,y,2] == 1.0f0) + 2*(input[x,y,3] == 1.0f0) + 3*(input[x,y,4] == 1.0f0) + 4*(input[x,y,5] == 1.0f0)

        if input[x, y, 6] == 1.0f0
            G.workers[G.playerToPlay, 1] = xy2cell(x,y)
        elseif input[x, y, 7] == 1.0f0
            G.workers[G.playerToPlay, 2] = xy2cell(x,y)
        end
        if input[x, y, 8] == 1.0f0
            G.workers[3-G.playerToPlay, idWorker2] = xy2cell(x,y)
            idWorker2 += 1
        end
    end
    !print_pos || print(G)

    vec = [x for x in enumerate(pi_MCTS)]
    sort!(vec, by=x->-x[2])
    for j in 1:5
        i,x = vec[j]
        (x > 0) || break
        println(round(Int,100*x), " %\t", get_printable_move(G, get_move_from_idx_move(G, i)))
    end
end



## Model 
struct SantoriniNN <: AbstractNN
    base::Chain
    trunk::Chain
    pi_head::Chain
    v_head::Chain
    name::String
    SantoriniNN() = new(
        Chain(Parallel(.+,
                Conv((3,3), SHAPE_INPUT[3]=>NUM_FILTERS, pad=1, bias=false),
                Chain(Dense(22=>NUM_FILTERS, relu, bias=false), Flux.unsqueeze(; dims=1), Flux.unsqueeze(; dims=2))
            ), BatchNorm(NUM_FILTERS, relu)) |> gpu,
        Chain(  [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:(NUM_LAYERS - 2 - 2*div(NUM_LAYERS-2,3))]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...) |> gpu,
        Chain(Conv((3,3), NUM_FILTERS=>PI_HEAD, pad=1, bias=false), BatchNorm(PI_HEAD, relu), Conv((3, 3), PI_HEAD => SHAPE_OUTPUT[3], pad=1), Flux.flatten) |> gpu, # pi
        Chain(Conv((3,3), NUM_FILTERS=>V_HEAD, pad=1, bias=false), BatchNorm(V_HEAD, relu), Flux.flatten, Dense(V_HEAD*25 => 1, sigmoid)) |> gpu, # v
        "b"*string(NUM_LAYERS)*"c"*string(NUM_FILTERS)*"_SE"
    )
    SantoriniNN(base, trunk, pi_head, v_head, name) = new(base, trunk, pi_head, v_head, name)
end

function NNet.predict(model::SantoriniNN, input_pos, input_params)
    @assert ndims(input_pos) == 4
    x = model.base(input_pos, input_params)
    y = model.trunk(x)
    return model.pi_head(y), model.v_head(y)
end
function NNet.predict(model::SantoriniNN, input)
    return predict(model, input...)
end

function NNet.predict(model::SantoriniNN, G::Santorini)
    input = add_last_dim(model, get_input_for_nn(G)) |> gpu
    output = predict(model, input) |> cpu
    piNN, vNN = squeeze(model, output)
    return piNN, vNN[1]
end

function NNet.get_input_shape(::SantoriniNN, batchsize::Int)
    return [zeros(Float32, (SHAPE_INPUT..., batchsize)), zeros(Float32, 22, batchsize)]    # pos, params
end
function NNet.get_output_shape(::SantoriniNN, batchsize::Int)
    return (zeros(Float32, NUMBER_ACTIONS, batchsize), zeros(Float32, 1, batchsize)) # pi, v
end

function NNet.concatenate_inputs(::SantoriniNN, inputs)
    # [ [input_pos, input_params], [input_pos, input_params], ... ] -> [inputs_pos, inputs_params]
    return [  
        [ input_pos[x, y, z] for x in 1:SHAPE_INPUT[1], y in 1:SHAPE_INPUT[2], z in 1:SHAPE_INPUT[3], (input_pos, _) in inputs ]  ,
        [input_params[i] for i in 1:22, (_, input_params) in inputs]
    ]
end

function NNet.inplace_change!(::SantoriniNN, inputs, i, input)
    # inputs[i] <- input it's made s.t. model.predict(inputs) is directly callable
    inputs[1][:, :, :, i] = input[1]
    inputs[2][:, i] = input[2]
end

function NNet.add_last_dim(::SantoriniNN, input) # go to 3dim to 4 dim (for convolutional networks)
    input_pos, input_params = input
    return [reshape(input_pos, size(input_pos)..., 1), reshape(input_params, size(input_params)..., 1)]
end
function NNet.squeeze(::SantoriniNN, output)
    return dropdims(output[1]; dims=2), dropdims(output[2]; dims=2)
end


function NNet.loss(model::SantoriniNN, input_pos, input_params, target)
    pi_nn, v_nn = predict(model, input_pos, input_params)
    Lv = Flux.mse(v_nn[1, :], target[NUMBER_ACTIONS+1, :])
    Lpi = Flux.Losses.logitcrossentropy(pi_nn, target[1:NUMBER_ACTIONS, :]; dims=1)   +    sum(  Flux.Losses.xlogx.(target[1:NUMBER_ACTIONS, :]) ) / size(target)[2]       #Kullback Leibler
    return Lv+Lpi, Lpi, Lv
end


function NNet.load(::Santorini, id)
    BSON.@load DIRECTORY*"model_"*string(id)*".bson" base trunk pi_head v_head name
    nn = SantoriniNN(base |> gpu, trunk |> gpu, pi_head |> gpu, v_head |> gpu, name)
    println("Model ", id, " loaded")
    return nn
end
function NNet.save(nn::SantoriniNN, id_model)
    base = nn.base |> cpu
    trunk = nn.trunk |> cpu
    pi_head = nn.pi_head |> cpu
    v_head = nn.v_head |> cpu
    name = nn.name
    BSON.@save DIRECTORY*"model_"*string(id_model)*".bson" base trunk pi_head v_head name
end
