const GAME = ARGS[1]  # choices : Santorini ; Azul ; Boop ; Resolve

if GAME == "Santorini"
    const DIRECTORY = "Santorini/"
    const CUCB = 0.25
    const HALFLIFE = 15
    const PARAMS_MCTS_ELO = Dict([200=>925, 400=>1138, 6400=>1640, 25600=>1879])

elseif GAME == "Boop"
    const DIRECTORY = "Boop/"
    const CUCB = 0.25
    const HALFLIFE = 30
    const PARAMS_MCTS_ELO = Dict([100=>570, 400=>1000, 1600 => 1629, 6400 => 2112])


elseif GAME == "Azul"
    const DIRECTORY = "Azul/"
    const CUCB = 0.25
    const HALFLIFE = 20.0
    const PARAMS_MCTS_ELO = Dict([100=>715, 400=>885, 1600 => 1054, 6400 => 1227])

elseif GAME == "Resolve"
    const DIRECTORY = "Resolve/"
    const CUCB = 1.5
    const HALFLIFE = 7.0
    const PARAMS_MCTS_ELO = Dict([100=>715, 400=>885, 1600 => 1054, 6400 => 1227])
else
    error("GAME not supported in params.jl : "*GAME)
end


const PARALLEL_RANDOM_NODES = 8
const GAME_SAVE_KEY = GAME
const DO_FORCED_PLAYOUTS = true
const kFP = 2.0
const FPU::Float32 = 0.2
const BASE_HALFLIFE = exp(-1.0/HALFLIFE)



# FOR THE LEARNING
# INIT. : do for first iterations until the neural has learnt some things (1h?)
#=
const LEARNING_PARAMS = Dict(
    "SIZE_REPLAYBUFFER" => 6000,
    "NGAMES" => 96,
    "lr_min" => 1e-4,
    "lr_max" => 1e-2,
    "MCTS_ITER" => 100,
    "BATCHSIZE" => 64,
    "WEIGHT_vMCTS" => 0.5,
    "EPOCHS" => 1,
    "WEIGHT_DIRICHLET" => 0.25,
    "EPISODES" => 300,
    "SAVE_EVERY" => 75,
)
const PCR_RATIO = 4 # every PCR_RATIO moves is not PCR (and thus exploratory)
const PCR_REDUCTION = 4
const TEMP_INIT = 2.00
const TEMP_FINAL = 1.00
const SURPRISE_WEIGHT = false
=#


# #MIDDLE a : after first iterations. Quality is poor and network definitely has to improve
#=
const LEARNING_PARAMS = Dict(
    "SIZE_REPLAYBUFFER" => 20000,
    "NGAMES" => 96,
    "lr_min" => 5e-5,
    "lr_max" => 5e-3,
    "MCTS_ITER" => 250,
    "BATCHSIZE" => 128,
    "WEIGHT_vMCTS" => 0.5,
    "EPOCHS" => 1,
    "WEIGHT_DIRICHLET" => 0.25,
    "EPISODES" => 500,
    "SAVE_EVERY" => 75,
)
const PCR_RATIO = 4 # every PCR_RATIO moves is not PCR (and thus exploratory)
const PCR_REDUCTION = 5
const TEMP_INIT = 1.00
const TEMP_FINAL = 0.20
const SURPRISE_WEIGHT = false
=#



# #MIDDLE b : neural network is good, and can still be better with a bigger batchsize.
const LEARNING_PARAMS = Dict(
    "SIZE_REPLAYBUFFER" => 80000,
    "NGAMES" => 96,
    "lr_min" => 5e-5,
    "lr_max" => 5e-3,
    "MCTS_ITER" => 400,
    "BATCHSIZE" => 128,
    "WEIGHT_vMCTS" => 0.5,
    "EPOCHS" => 1,
    "WEIGHT_DIRICHLET" => 0.25,
    "EPISODES" => 500,
    "SAVE_EVERY" => 75,
)
const PCR_RATIO = 4 # every PCR_RATIO moves is not PCR (and thus exploratory)
const PCR_REDUCTION = 5
const TEMP_INIT = 1.00
const TEMP_FINAL = 0.20
const SURPRISE_WEIGHT = true

# END : network is very good and will learn from a learning with 1600 iterations
#=
const LEARNING_PARAMS = Dict(
    "SIZE_REPLAYBUFFER" => 400000,
    "NGAMES" => 196,
    "lr_min" => 5e-5,
    "lr_max" => 5e-3,
    "MCTS_ITER" => 1600,
    "BATCHSIZE" => 128,
    "WEIGHT_vMCTS" => 0.5,
    "EPOCHS" => 1,
    "WEIGHT_DIRICHLET" => 0.25,
    "EPISODES" => 500,
    "SAVE_EVERY" => 10,
)
const PCR_RATIO = 5 # every PCR_RATIO moves is not PCR (and thus exploratory)
const PCR_REDUCTION = 8 # when in PCR, do MCTS_ITER / PCR_REDUCTION iterations only
const TEMP_INIT = 1.00
const TEMP_FINAL = 0.20
const SURPRISE_WEIGHT = true
=#





# OTHER CONSTANTS
const INF = 1000000.0
const TERMINAL_WIN = 3
const TERMINAL_DRAW = 2
const TERMINAL_LOSS = 1
const NOT_TERMINAL = 0
