using Flux, Random


#For NN
const NN_TYPE = "Conv"
const NUM_FILTERS = 196
const NUM_LAYERS = 15
const V_HEAD = 128
const PI_HEAD = 128
const SHAPE_OUTPUT = (36, 1, 1)
const NUMBER_OUTPUTS = 42
const NUMBER_ACTIONS = 36     #Total number of actions (=size of pi_output)
const LENGTH_POINTS = 5


function Game.load_from(Gfrom::Azul, Gto::Azul)
    Gto.board .= Gfrom.board
    Gto.pattern .= Gfrom.pattern   #does a copy
    Gto.floor .= Gfrom.floor
    Gto.scores .= Gfrom.scores
    Gto.hasFirstPlayerTile .= Gfrom.hasFirstPlayerTile
    Gto.factory .= Gfrom.factory
    Gto.buffer .= Gfrom.buffer
    Gto.playerToPlay = Gfrom.playerToPlay
    Gto.finished = Gfrom.finished
    Gto.round = Gfrom.round
    Gto.bag .= Gfrom.bag
    Gto.core_rng = copy(Gfrom.core_rng)
    Random.seed!(Gto.core_rng, rand(Int))
end



function Game.get_idx_output(G::Azul, move::AzulMove) :: Int
    if move.kind == ACTION_TAKE
        return move.color + 5*move.line - 5 #(1,1) to (5,CENTER)
    else
        return 5*CENTER + move.line   #line 1 to 6
    end
end
# After output of nn, we chose move number idx. What move is it corresponding to ?
function Game.get_move_from_idx_move(G::Azul, idx::Int) :: AzulMove
    if idx > 5*CENTER
        return AzulMove(ACTION_PLACE, idx-5*CENTER, findmax(G.buffer)[2], 0)
    else
        return AzulMove(ACTION_TAKE, div(idx-1, 5)+1, mod(idx-1,5)+1, 0)
    end
end

function get_filter_pattern(patterns)
    filter = zeros(Float32, 5, 5)
    for line in 1:5
        qty, color = patterns[line]
        if qty == 0
            continue
        end
        filter[line, COLORPOS[line, color]] = qty / line
    end
    return filter
end
function get_filter(bag)
    filter = zeros(Float32, 5, 5)
    for line in 1:5
        for color in 1:5
            filter[line, COLORPOS[line, color]] = bag[color]
        end
    end
    return filter
end
function get_bag(filter)
    bag = zeros(Int, 5)
    for color = 1:5
        bag[color] = round(Int, filter[1, COLORPOS[1, color]])
    end
    return bag
end

function Game.get_input_for_nn(G::Azul)
    input_playera_pos    = zeros(Float32, 5, 5, 2)
    #input_playera_params = zeros(Float32, 3)
    input_playerb_pos    = zeros(Float32, 5, 5, 2)
    #input_playerb_params = zeros(Float32, 3)
    remaining    = zeros(Float32, 5, 5, 8)

    if G.playerToPlay == 0
        print(G)
    end
    @assert G.playerToPlay != 0

    #state of both players
    p = G.playerToPlay
    input_playera_pos[:, :, 1] = G.board[:, :, p]
    input_playera_pos[:, :, 2] .= get_filter_pattern(G.pattern[p, :])
    input_playera_params = Float32[G.scores[p] / 10.0, min(G.floor[p], 7)/7.0, G.hasFirstPlayerTile[p]]

    input_playerb_pos[:, :, 1] = G.board[:, :, 3-p]
    input_playerb_pos[:, :, 2] .= get_filter_pattern(G.pattern[3-p, :])
    input_playerb_params = Float32[G.scores[3-p] / 10.0, min(G.floor[3-p], 7)/7.0, G.hasFirstPlayerTile[3-p]]


    # remaining :
    # 6 factories
    # 1 buffer
    # 1 bag

    for i in 1:CENTER
        remaining[:, :,   i] .= get_filter(G.factory[i, :]) / 4.0
    end
    remaining[:, :, 7] = get_filter(G.buffer) / 5.0
    remaining[:, :, 8] = get_filter(G.bag) / 20.0

    return [input_playera_pos, input_playera_params, input_playerb_pos, input_playerb_params, remaining]
end

function Game.get_input_for_nn!(G::Azul, pi_MCTS::Vector{Float32}, rng)
    input = get_input_for_nn(G)
    
    if sum(G.buffer) == 0
        #switch factories
        s = Random.shuffle(rng, [1,2,3,4,5])
        input[5][:, :, 1:5] = input[5][:, :, s]
        pi_MCTS_save = copy(pi_MCTS)
        # In piMCTS, factory i is contained in (1:5) .+ (5*i - 5)
        for f in 1:5
            pi_MCTS[(1:5) .+ (5*f-5)] .= pi_MCTS_save[(1:5) .+ (5*s[f]-5)]
        end
    end

    return input
end



function Game.print_input_nn(::Azul, input, pi_MCTS; print_pos=true)
    G = new_game()
    input_playera_pos, input_playera_params, input_playerb_pos, input_playerb_params, remaining = input

    for x in 1:5, y in 1:5
        G.board[x,y,1] = round(Bool, input_playera_pos[x, y, 1])
        G.board[x,y,2] = round(Bool, input_playerb_pos[x, y, 1])
    end
    for line in 1:5, y in 1:5
        if input_playera_pos[line, y, 2] > 0.0f0
            qty = input_playera_pos[line, y, 2] * line
            color = BOARDCOLOR[line, y]
            G.pattern[1, line] = (qty, color)
        end
        if input_playerb_pos[line, y, 2] > 0.0f0
            qty = input_playerb_pos[line, y, 2] * line
            color = BOARDCOLOR[line, y]
            G.pattern[2, line] = (qty, color)
        end
    end
    G.scores[1] = round(Int, input_playera_params[1]*10)
    G.floor[1] = round(Int, input_playera_params[2]*7)
    G.hasFirstPlayerTile[1] = round(Bool, input_playera_params[3])
    G.scores[2] = round(Int, input_playerb_params[1]*10)
    G.floor[2] = round(Int, input_playerb_params[2]*7)
    G.hasFirstPlayerTile[2] = round(Bool, input_playerb_params[3])


    for f in 1:CENTER
        G.factory[f, :] .= get_bag(remaining[:, :, f] * 4.0)
    end
    G.buffer .= get_bag(remaining[:, :, 7]*5.0)
    G.bag = get_bag(remaining[:, :, 8]*20.0)

    if print_pos
        print(G)
    end

    moves = [(-pi_MCTS[idx], idx) for idx in 1:(6+CENTER*5)]
    sort!(moves)
    for i in 1:5
        if moves[i][1] == 0
            break
        end
        println(round(Int, -100*moves[i][1]), "%\t", get_printable_move(  G, get_move_from_idx_move(G, moves[i][2])) )

    end

end




## Model 
struct AzulNN <: AbstractNN
    player_base::Chain
    factories_base::Chain
    trunk::Chain
    pi_head::Chain
    v_head::Chain
    points_head::Chain
    name::String
    function AzulNN()
        filters_player, filters_factories = div(NUM_FILTERS, 3), NUM_FILTERS - 2*div(NUM_FILTERS, 3)
        return new(Chain(Parallel(.+,
                Conv((3,3), 2=>filters_player, pad=1, bias=false),
                Chain(Dense(3=>filters_player, relu), Flux.unsqueeze(; dims=1), Flux.unsqueeze(; dims=2))
            ), BatchNorm(filters_player, relu)) |> gpu,
            Chain(Conv((3,3), 8 => filters_factories, pad=1, bias=false), BatchNorm(filters_factories, relu)) |> gpu,
            Chain([ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...,
                    SEResNetBlock(),
                    [ResNetBlock() for _ in 1:(NUM_LAYERS - 2 - 2*div(NUM_LAYERS-2,3))]...,
                    SEResNetBlock(),
                    [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...) |> gpu,
            Chain(Conv((3,3), NUM_FILTERS=>PI_HEAD, pad=1, bias=false), BatchNorm(PI_HEAD, relu), Flux.flatten, Dense(PI_HEAD*25 => NUMBER_ACTIONS)) |> gpu, # pi
            Chain(Conv((3,3), NUM_FILTERS=>V_HEAD, pad=1, bias=false), BatchNorm(V_HEAD, relu), Flux.flatten, Dense(V_HEAD*25 => 1, sigmoid)) |> gpu, # v
            Chain(Conv((3,3), NUM_FILTERS=>V_HEAD, pad=1, bias=false), BatchNorm(V_HEAD, relu), Flux.flatten, Dense(V_HEAD*25 => 5)) |> gpu, # points
            "b"*string(NUM_LAYERS)*"c"*string(NUM_FILTERS)*"_SE")
    end
    AzulNN(pb, fb, trunk, pi_head, v_head, points_head, name) = new(pb, fb, trunk, pi_head, v_head, points_head, name)
end


function NNet.predict(model::AzulNN, inputs)
    playera_pos, playera_params, playerb_pos, playerb_params, remaining = inputs

    #playera
    xa = model.player_base(playera_pos, playera_params)

    #playerb
    xb = model.player_base(playerb_pos, playerb_params)

    #remaining
    xrem = model.factories_base(remaining)

    y = model.trunk( cat(xa, xb, xrem; dims=3) )
    return model.pi_head(y), model.v_head(y), model.points_head(y)
end

function NNet.predict(model::AzulNN, G::Azul)
    input = add_last_dim(model, get_input_for_nn(G)) |> gpu
    output = predict(model, input) |> cpu
    piNN, vNN, pointsNN = squeeze(model, output)
    return piNN, vNN[1], pointsNN
end


function NNet.get_input_shape(::AzulNN, batchsize::Int)
    return [zeros(Float32, 5, 5, 2, batchsize), zeros(Float32, 3, batchsize), zeros(Float32, 5, 5, 2, batchsize), zeros(Float32, 3, batchsize), zeros(Float32, 5, 5, 8, batchsize)]
end
function NNet.get_output_shape(::AzulNN, batchsize::Int)
    return [zeros(Float32, NUMBER_ACTIONS, batchsize), zeros(Float32, 1, batchsize), zeros(Float32, 5, batchsize)] # pi, v, points
end

function NNet.concatenate_inputs(::AzulNN, inputs)
    return [  
        [player_posa[x, y, z] for x in 1:5, y in 1:5, z in 1:2, (player_posa, _, _, _, _) in inputs ]  ,
        [params_playera[i] for i in 1:3, (_, params_playera, _, _, _) in inputs],
        [player_posb[x, y, z] for x in 1:5, y in 1:5, z in 1:2, (_, _, player_posb, _, _) in inputs ]  ,
        [params_playerb[i] for i in 1:3, (_, _, _, params_playerb, _) in inputs],
        [remaining[x, y, z] for x in 1:5, y in 1:5, z in 1:8, (_, _, _, _, remaining) in inputs]
    ]
end

function NNet.inplace_change!(::AzulNN, inputs, i, input)
    # inputs[i] <- input it's made s.t. model.predict(inputs) is directly callable
    inputs[1][:,:,:,i] = input[1]
    inputs[2][:, i] = input[2]
    inputs[3][:,:,:,i] = input[3]
    inputs[4][:, i] = input[4]
    inputs[5][:,:,:,i] = input[5]
end

function NNet.add_last_dim(::AzulNN, inputs) # go to 3dim to 4 dim (for convolutional networks)
    playera_pos, playera_params, playerb_pos, playerb_params, remaining = inputs
    return [
        reshape(playera_pos, size(playera_pos)..., 1), 
        reshape(playera_params, size(playera_params)..., 1), 
        reshape(playerb_pos, size(playerb_pos)..., 1), 
        reshape(playerb_params, size(playerb_params)..., 1), 
        reshape(remaining, size(remaining)..., 1)
    ]
end
function NNet.squeeze(::AzulNN, output)
    return dropdims(output[1]; dims=2), dropdims(output[2]; dims=2), dropdims(output[3]; dims=2)
end


function NNet.loss(model::AzulNN, input_posa, input_pa, input_posb, input_pb, inoput_remaining, target)
    pi_nn, v_nn, points_nn = predict(model, [input_posa, input_pa, input_posb, input_pb, inoput_remaining])
    Lpi = Flux.Losses.logitcrossentropy(pi_nn, target[1:NUMBER_ACTIONS, :]; dims=1)   +    sum(  Flux.Losses.xlogx.(target[1:NUMBER_ACTIONS, :]) ) / size(target)[2]       #Kullback Leibler
    Lv = Flux.mse(v_nn[1, :], target[NUMBER_ACTIONS+1, :])
    Lpoints = 0.02 * (  Flux.mse(points_nn[1, :], target[NUMBER_ACTIONS+2, :])/20  +
                        Flux.mse(points_nn[2, :], target[NUMBER_ACTIONS+3, :])/100  +
                        Flux.mse(points_nn[3, :], target[NUMBER_ACTIONS+4, :]) + 
                        Flux.mse(points_nn[4, :], target[NUMBER_ACTIONS+5, :])*2 + 
                        Flux.mse(points_nn[5, :], target[NUMBER_ACTIONS+6, :])*6 )
    return Lv+Lpi+Lpoints, Lpi, Lv, Lpoints
end

function NNet.load(::Azul, id)
    BSON.@load DIRECTORY*"model_"*string(id)*".bson" pb fb trunk pi_head v_head points_head name
    nn = AzulNN(pb |> gpu, fb |> gpu, trunk |> gpu, pi_head |> gpu, v_head |> gpu, points_head |> gpu, name)
    println("Model ", id, " loaded")
    return nn
end
function NNet.save(nn::AzulNN, id_model)
    pb = nn.player_base |> cpu
    fb = nn.factories_base |> cpu
    trunk = nn.trunk |> cpu
    pi_head = nn.pi_head |> cpu
    v_head = nn.v_head |> cpu
    points_head = nn.points_head |> cpu
    name = nn.name
    BSON.@save DIRECTORY*"model_"*string(id_model)*".bson" pb fb trunk pi_head v_head points_head name
end
