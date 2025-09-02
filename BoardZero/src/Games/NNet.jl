module NNet

export predict, loss, get_input_shape, get_output_shape, concatenate_inputs, add_last_dim, squeeze
export initialize_model, are_all_nn_set
export AbstractNN

include("../params.jl")

using BSON
using ..Game

abstract type AbstractNN end

function predict(::AbstractNN, d...)
    error("No function for abstractNN")
end

function loss(::AbstractNN, d...)
    error("No loss function for abstractNN")
end

function load(::AbstractGame, id)
    error("No function load for abstractNN")
end
function save(::AbstractNN)
    error("No function save for AbstractNN")
end

function get_input_shape(::AbstractNN, batchsize::Int)
    error("No function get_input_shape for abstractnn")
end
function get_output_shape(::AbstractNN, batchsize::Int)
    error("No function get_output_shape for abstractnn")
end

function inplace_change!(::AbstractNN, inputs, i, input)
    # inputs[i] <- input it's made s.t. model.predict(inputs) is directly callable
    error("No function inplace_change! for abstractnn")
end


function concatenate_inputs(::AbstractNN, inputs)
    #Let's say all inputs have 2 elements. Then it transforms :
    #inputs = [[1st input which is [A, B]], [2nd input], [3rd input] ...]
    #Into :
    #[[concatenation of 1st element of As], [concatenation of 2nd element of Bs]]
end
function add_last_dim(::AbstractNN, input) # go to 3dim to 4 dim (for convolutional networks)
    error("No function drop_last_dim for abstractnn")
end
function squeeze(::AbstractNN, output)
    error("No function squeeze for abstractnn")
end



function initialize_model(arg) :: AbstractNN
    if !isfile(DIRECTORY*"model_1.bson") || arg == :new
        # B) BUILD
        println("New model being created...")
        model = new_model()
        return model

    elseif arg == :last
        #LOAD
        id = 1
        while isfile(DIRECTORY*"model_"*string(id)*".bson")
            id += 1
        end
        return initialize_model(id-1)

    elseif arg == :ELO
        bestELO = -1000
        bestid = 0
        id = 1
        while isfile(DIRECTORY*"stats_"*string(id)*".bson")
            d = BSON.load(DIRECTORY*"stats_"*string(id)*".bson")
            if d[:ELO] >= bestELO
                bestELO = d[:ELO]
                bestid = id
            end
            id += 1
        end
        @assert bestid != 0
        return initialize_model(bestid)

    elseif typeof(arg) == Int    #id == arg
        return load(BLANK_GAME, arg)

    elseif typeof(arg) == String
        BSON.@load arg nn_cpu
        nn = nn_cpu |> DEVICE
        println("Model ", arg, " loaded")
        return nn

    else
        error("Type of argument")

    end
end

function are_all_nn_set()
    id = 1
    while isfile(DIRECTORY*"stats_"*string(id)*".bson")
        d = BSON.load(DIRECTORY*"stats_"*string(id)*".bson")
        if d[:ELO] == -1
            return false
        end
        id += 1
    end
    return true
end


function ResNetBlock()
    return Chain(
        SkipConnection(Chain(
            Conv((3,3), NUM_FILTERS=>NUM_FILTERS, pad=1, bias=false),
            BatchNorm(NUM_FILTERS, relu),
            Conv((3,3), NUM_FILTERS=>NUM_FILTERS, pad=1, bias=false),
            BatchNorm(NUM_FILTERS),
            ), +
        ),
        relu
    )
end

adaptivemeanpool(x) = sum(sum(x; dims=1); dims=2) / NUM_CELLS

function squeeze_excite()
    return SkipConnection( Chain(
            Flux.GlobalMeanPool(),
            Flux.flatten,
            Dense(NUM_FILTERS => div(NUM_FILTERS , 2), relu),
            Dense(div(NUM_FILTERS , 2) => NUM_FILTERS, sigmoid),
            Flux.unsqueeze(; dims=1),
            Flux.unsqueeze(; dims=2) # it's now 1x1xNUMFILTERSx1
        ), .*
    )
end

function SEResNetBlock()
    return Chain(
    SkipConnection(Chain(
        Conv((3,3), NUM_FILTERS=>NUM_FILTERS, pad=1, bias=false),
        BatchNorm(NUM_FILTERS, relu),
        Conv((3,3), NUM_FILTERS=>NUM_FILTERS, pad=1, bias=false),
        BatchNorm(NUM_FILTERS),
        squeeze_excite()
    ), +),
    relu)
end


@static if GAME == "Santorini"
    include("Santorini/SantoriniNN.jl")
    export SantoriniNN
elseif GAME == "Boop"
    include("Boop/BoopNN.jl")
    export BoopNN
elseif GAME == "Resolve"
    include("Resolve/ResolveNN.jl")
    export ResolveNN
elseif GAME == "Azul"
    include("Azul/AzulNN.jl")
    export ResolveNN
end
export NUMBER_ACTIONS, LENGTH_POINTS

function new_model()
    @static if GAME == "Santorini"
        return SantoriniNN()
    elseif GAME == "Boop"
        return BoopNN()
    elseif GAME == "Resolve"
        return ResolveNN()
    elseif GAME == "Azul"
        return AzulNN()
    else
        error("Game not available !")
    end
end

end