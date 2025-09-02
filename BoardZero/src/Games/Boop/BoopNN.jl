using Flux, Crayons

#For NN
const NN_TYPE = "Conv"
const NUM_FILTERS = 128
const NUM_LAYERS = 10
const V_HEAD = 32
const PI_HEAD = 32
const LENGTH_POINTS = 0
const SHAPE_INPUT = (6, 6, 4)   # plus 4 scalar parameters (numbers of kittens cats of each player)
const NUM_CELLS = 36
const SHAPE_OUTPUT = (6, 6, 2)    #2 : cat play and kitten play
const NUMBER_ACTIONS = 72     #Total number of actions (=size of pi_output)

function Game.load_from(Gfrom::Boop, Gto::Boop)
    Gto.kittens .= Gfrom.kittens
    Gto.cats .= Gfrom.cats
    Gto.playerToPlay = Gfrom.playerToPlay
    Gto.remaining_kittens .= Gfrom.remaining_kittens
    Gto.remaining_cats .= Gfrom.remaining_cats
    Gto.finished = Gfrom.finished
    Gto.ntour = Gfrom.ntour
end


function Game.get_input_for_nn(G::Boop)
    input::Array{Float32, 3} = zeros(Float32, 6, 6, 4)
    p = G.playerToPlay
    for x in 1:6, y in 1:6
        cell = cells[x + y*6 - 6]
        if G.kittens[p] & cell > 0
            input[x, y, 1] = 1.0f0
        elseif G.cats[p] & cell > 0
            input[x, y, 2] = 1.0f0
        elseif G.kittens[3-p] & cell > 0
            input[x, y, 3] = 1.0f0
        elseif G.cats[3-p] & cell > 0
            input[x, y, 4] = 1.0f0
        end
    end
    input_params = Float32[G.remaining_kittens[p]/8, G.remaining_cats[p]/8, G.remaining_kittens[3-p]/8, G.remaining_cats[3-p]/8]

    return [input, input_params]
end

function Game.get_input_for_nn!(G::Boop, pi_MCTS::Vector{Float32}, rng)
    input, input_params = get_input_for_nn(G)
    pi_MCTS = reshape(pi_MCTS, 6, 6, 2)

    if rand(rng, Bool)
        input[:, :, 1:4] = reverse(input[:, :, 1:4]; dims=2)
        pi_MCTS .= reverse(pi_MCTS; dims=2)
    end

    r = rand(rng, 1:4)
    if r == 1
        for i in 1:4
            input[:, :, i] = rotr90(input[:, :, i])
        end
        pi_MCTS[:, :, 1] .= rotr90(pi_MCTS[:, :, 1])
        pi_MCTS[:, :, 2] .= rotr90(pi_MCTS[:, :, 2])

    elseif r == 2
        for i in 1:4
            input[:, :, i] = rot180(input[:, :, i])
        end
        pi_MCTS[:, :, 1] .= rot180(pi_MCTS[:, :, 1])
        pi_MCTS[:, :, 2] .= rot180(pi_MCTS[:, :, 2])

    elseif r == 3
        for i in 1:4
            input[:, :, i] = rotl90(input[:, :, i])
        end
        pi_MCTS[:, :, 1] .= rotl90(pi_MCTS[:, :, 1])
        pi_MCTS[:, :, 2] .= rotl90(pi_MCTS[:, :, 2])

    end

    pi_MCTS = reshape(pi_MCTS, :)
    return [input, input_params]
end


function Game.get_idx_output(G::Boop, move::BoopMove) :: Int
    return move.c + 36*move.cat
end

function Game.print_input_nn(::Boop, input, pi_MCTS; print_pos=true)
    input_pos, input_params = input
    println("Rem. : ", round(Int, 8*input_params[1]), CHAR[2][1], " ", round(Int, 8*input_params[2]), CHAR[2][2], "\tRem. : ", round(Int, 8*input_params[3]), CHAR[1][1], " ", round(Int, 8*input_params[4]), CHAR[1][2])
    f(x) = round(Int, 100*x)
    text(x) = (f(x) == 100) ? "100 " : ((f(x) >= 10) ? string(f(x))*"  " : string(f(x))*"   ")
    for x in 1:6
        for y in 1:6
            if print_pos
                if input_pos[x, y, 1] > 0
                    print(crayon"yellow", " k ", crayon"reset")
                elseif input_pos[x, y, 2] > 0
                    print(crayon"yellow", " C ", crayon"reset")
                elseif input_pos[x, y, 3] > 0
                    print(crayon"blue", " k ", crayon"reset")
                elseif input_pos[x, y, 4] > 0
                    print(crayon"blue", " C ", crayon"reset")
                else
                    print(" . ")
                end
            end
        end
        print("\t")
        for y in 1:6
            cell = x+y*6-6
            print(text(pi_MCTS[cell]))  #placing a kitten
        end
        print("\t")
        for y in 1:6
            cell = x+y*6-6+36
            print(text(pi_MCTS[cell]))  #placing a cat
        end
        println()
    end
end


# Model methods :

## Model 
struct BoopNN <: AbstractNN
    #filters_base::Chain     # For positions of kittens (me), cats (me), kittens (he), cats (he) 
    #parameters_base::Dense  # for number of remaining kittens, cats, for both players
    base::Chain # (input_pos, input_params) -> NUM_FILTERS filters
    trunk::Chain
    pi_head::Chain
    v_head::Chain
    name::String
    BoopNN() = new(
        Chain(Parallel(.+,
                Conv((3,3), SHAPE_INPUT[3]=>NUM_FILTERS, pad=1, bias=false),
                Chain(Dense(4=>NUM_FILTERS, relu, bias=false), Flux.unsqueeze(; dims=1), Flux.unsqueeze(; dims=2))
            ), BatchNorm(NUM_FILTERS, relu)) |> gpu,
        Chain(
                [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:(NUM_LAYERS - 2 - 2*div(NUM_LAYERS-2,3))]...,
                SEResNetBlock(),
                [ResNetBlock() for _ in 1:div(NUM_LAYERS-2,3)]...) |> gpu,
        #Chain(Conv((3,3), NUM_FILTERS=>PI_HEAD, pad=1, bias=false), BatchNorm(PI_HEAD, relu), Flux.flatten, Dense(PI_HEAD*36 => NUMBER_ACTIONS)) |> gpu, # pi
        Chain(Conv((3,3), NUM_FILTERS=>PI_HEAD, pad=1, bias=false), BatchNorm(PI_HEAD, relu), Conv((3,3), PI_HEAD=>SHAPE_OUTPUT[3], pad=1), Flux.flatten) |> gpu, # pi
        Chain(Conv((3,3), NUM_FILTERS=>V_HEAD, pad=1, bias=false), BatchNorm(V_HEAD, relu), Flux.flatten, Dense(V_HEAD*36 => 1, sigmoid)) |> gpu, # v
        "b"*string(NUM_LAYERS)*"c"*string(NUM_FILTERS)*"_SE"
    )
    BoopNN(base, trunk, pi_head, v_head, name) = new(base, trunk, pi_head, v_head, name)
end

function NNet.predict(model::BoopNN, input_pos, input_params)
    @assert ndims(input_pos) == 4
    x = model.base(input_pos, input_params)
    y = model.trunk(x)
    return model.pi_head(y), model.v_head(y)
end
function NNet.predict(model::BoopNN, input)
    return predict(model, input...)
end

function NNet.predict(model::BoopNN, G::Boop)
    input = add_last_dim(model, get_input_for_nn(G)) |> gpu
    output = predict(model, input) |> cpu
    piNN, vNN = squeeze(model, output)
    return piNN, vNN[1]
end

function NNet.get_input_shape(::BoopNN, batchsize::Int)
    return [zeros(Float32, (SHAPE_INPUT..., batchsize)), zeros(Float32, 4, batchsize)]    # pos, params
end
function NNet.get_output_shape(::BoopNN, batchsize::Int)
    return [zeros(Float32, NUMBER_ACTIONS, batchsize), zeros(Float32, 1, batchsize)] # pi, v
end

function NNet.concatenate_inputs(::BoopNN, inputs)
    # [ [input_pos, input_params], [input_pos, input_params], ... ] -> [inputs_pos, inputs_params]
    return [  
        [ input_pos[x, y, z] for x in 1:SHAPE_INPUT[1], y in 1:SHAPE_INPUT[2], z in 1:SHAPE_INPUT[3], (input_pos, _) in inputs ]  ,
        [input_params[i] for i in 1:4, (_, input_params) in inputs]
    ]
end

function NNet.inplace_change!(::BoopNN, inputs, i, input)
    # inputs[i] <- input it's made s.t. model.predict(inputs) is directly callable
    inputs[1][:,:,:,i] = input[1]
    inputs[2][:, i] = input[2]
end

function NNet.add_last_dim(::BoopNN, input) # go to 3dim to 4 dim (for convolutional networks)
    input_pos, input_params = input
    return [reshape(input_pos, size(input_pos)..., 1), reshape(input_params, size(input_params)..., 1)]
end
function NNet.squeeze(::BoopNN, output)
    return dropdims(output[1]; dims=2), dropdims(output[2]; dims=2)
end

function NNet.loss(model::BoopNN, input_pos, input_params, target)
    pi_nn, v_nn = predict(model, input_pos, input_params)
    Lv = Flux.mse(v_nn[1, :], target[NUMBER_ACTIONS+1, :])
    Lpi = Flux.Losses.logitcrossentropy(pi_nn, target[1:NUMBER_ACTIONS, :]; dims=1)   +    sum(  Flux.Losses.xlogx.(target[1:NUMBER_ACTIONS, :]) ) / size(target)[2]       #Kullback Leibler
    return Lv+Lpi, Lpi, Lv
end


function NNet.load(::Boop, id)
    BSON.@load DIRECTORY*"model_"*string(id)*".bson" base trunk pi_head v_head name
    nn = BoopNN(base |> gpu, trunk |> gpu, pi_head |> gpu, v_head |> gpu, name)
    println("Model ", id, " loaded")
    return nn
end
function NNet.save(nn::BoopNN, id_model)
    base = nn.base |> cpu
    trunk = nn.trunk |> cpu
    pi_head = nn.pi_head |> cpu
    v_head = nn.v_head |> cpu
    name = nn.name
    BSON.@save DIRECTORY*"model_"*string(id_model)*".bson" base trunk pi_head v_head name
end
