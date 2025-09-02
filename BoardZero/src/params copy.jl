using Flux: gpu, cpu

const GAME = "Santorini"  # choices : Gomoku6 ; Gomoku9 ; Gomoku13 ; Gomoku15 ; Connect4 ; Pente ; Santorini ; Kalah ; Boop ; Jaipur ; Resolve

if GAME == "Gomoku6"
    #For the game
    const SIZE = 6
    const SIZE2 = SIZE*SIZE
    const ALIGN = 4
    const NUM_PLAYERS = 2

    const DIRECTORY = "664/"

    # For MCTS
    const CUCB = 0.5 # For Gomoku6 on 15000-MCTS better than 0.3 (+33ELO)
    const HALFLIFE = 6


    # For a neural network
    const NN_TYPE = "Conv"
    const NUM_FILTERS = 128
    const NUM_LAYERS = 4
    const V_HEAD = 4
    const PI_HEAD = 8
    const WIDTH = SIZE
    const LENGTH = WIDTH
    const SHAPE_BOARD = (WIDTH, LENGTH)
    const SHAPE_INPUT = (WIDTH, LENGTH, 2)
    const NUM_CELLS = WIDTH*LENGTH
    const SHAPE_OUTPUT = (WIDTH, LENGTH, 1)    #One proba per position
    const NUMBER_ACTIONS = WIDTH*LENGTH     #Total number of actions (=size of pi_output)

    const PARAMS_MCTS_ELO = Dict([2=>0, 50=>604, 400=>1000, 800=>1131, 1600=>1275, 6400=>1411, 12800=>1471])  # for 664
#     const PARAMS_MCTS_ELO = Dict([2=>604, 20=>1000, 100=>1131])  # for 664


elseif GAME == "Gomoku9"

    #FOR THE GAME
    const SIZE = 9
    const SIZE2 = SIZE*SIZE
    const ALIGN = 5
    const NUM_PLAYERS = 2

    const DIRECTORY = "995/"

    # For MCTS
    const CUCB = 0.25 # For Gomoku9 on 100 & 1000-MCTS matches +21ELO vs 0.3 & +14ELO vs 0.2
    const HALFLIFE = 10 # In training, in each game temperature decreases at this rate : it halves every HALFLIFE moves


    #For NN
    const NN_TYPE = "Conv"
    const NUM_FILTERS = 128
    const NUM_LAYERS = 10
    const V_HEAD = 16
    const PI_HEAD = 16
    const WIDTH = SIZE
    const LENGTH = WIDTH
    const SHAPE_BOARD = (WIDTH, LENGTH)
    const SHAPE_INPUT = (WIDTH, LENGTH, 2)
    const NUM_CELLS = WIDTH*LENGTH
    const SHAPE_OUTPUT = (WIDTH, LENGTH, 1)    #One proba per position
    const NUMBER_OUTPUTS = 9*9+1
    const NUMBER_ACTIONS = WIDTH*LENGTH     #Total number of actions (=size of pi_output)

    #const PARAMS_MCTS_ELO = Dict([50=>641, 400=>1000, 800=>1162, 1600=>1412, 6400=>1788, 12800=>1965])  # for 995
    const PARAMS_MCTS_ELO = Dict([6400=>1788, 12800=>1965, 25600=>2092])   # for 995


elseif GAME == "Gomoku13"
    #For the game
    const SIZE = 13
    const SIZE2 = SIZE*SIZE
    const ALIGN = 5
    const NUM_PLAYERS = 2

    const DIRECTORY = "Gomoku13/"
    #const CUCB = 0.2 # For Gomoku11 on 100 & 1000-MCTS matches, +40ELO vs 0.15 & +32ELO vs 0.25
    const CUCB = 0.25 # For Gomoku13
    const HALFLIFE = 12

    #For NN
    const NN_TYPE = "Conv"
    const NUM_FILTERS = 128
    const NUM_LAYERS = 10
    const V_HEAD = 8
    const PI_HEAD = 16
    const WIDTH = SIZE
    const LENGTH = WIDTH
    const SHAPE_BOARD = (WIDTH, LENGTH)
    const SHAPE_INPUT = (WIDTH, LENGTH, 2)
    const NUM_CELLS = WIDTH*LENGTH
    
    const SHAPE_OUTPUT = (WIDTH, LENGTH, 1)    #One proba per position
    const NUMBER_ACTIONS = WIDTH*LENGTH     #Total number of actions (=size of pi_output)

        #const PARAMS_MCTS_ELO = Dict([50=>667, 400=>1000, 1600=>1279, 6400=>1668])  # for Gomoku13
    #const PARAMS_MCTS_ELO = Dict([6400=>1668, 12800=>1851]) # for Gomoku13
    #const PARAMS_MCTS_ELO = Dict([12800=>1851, 25600=>2051]) # for Gomoku13
    #const PARAMS_MCTS_ELO = Dict([25600=>2051, 50000=>2207]) # for Gomoku13

elseif GAME == "Gomoku15"
    #For the game
    const SIZE = 15
    const SIZE2 = SIZE*SIZE
    const ALIGN = 5
    const NUM_PLAYERS = 2

    const DIRECTORY = "Gomoku15/"
    #const CUCB = 0.2 # For Gomoku11 on 100 & 1000-MCTS matches, +40ELO vs 0.15 & +32ELO vs 0.25
    const CUCB = 0.25 # For Gomoku13
    const HALFLIFE = 12

    #For NN
    const NN_TYPE = "Conv"
    const NUM_FILTERS = 128
    const NUM_LAYERS = 10
    const V_HEAD = 8
    const PI_HEAD = 16
    const WIDTH = SIZE
    const LENGTH = WIDTH
    const SHAPE_BOARD = (WIDTH, LENGTH)
    const SHAPE_INPUT = (WIDTH, LENGTH, 2)
    const NUM_CELLS = WIDTH*LENGTH
    const SHAPE_OUTPUT = (WIDTH, LENGTH, 1)    #One proba per position
    const NUMBER_OUTPUTS = 15*15+1
    const NUMBER_ACTIONS = WIDTH*LENGTH     #Total number of actions (=size of pi_output)

    const PARAMS_MCTS_ELO = Dict([400=>710, 1600=>1047, 3200=>1187])  # for Gomoku15



elseif GAME == "Pente"
    const WIDTH = 9
    const LENGTH = WIDTH
    const SHAPE_INPUT = (WIDTH, LENGTH, 4)   # 2 canaux pour la position du prochain joueur et de son adversaire ; et 2 canaux de ones*score de chacun des joueurs.
    const SHAPE_BOARD = (WIDTH, LENGTH)
    const NUM_CELLS = WIDTH*LENGTH
    const SHAPE_OUTPUT = (WIDTH, LENGTH, 1)
    const NUMBER_ACTIONS = WIDTH*LENGTH

    const DIRECTORY = "Pente/"

    #FOR THE GAME
    const SIZE = 9
    const SIZE2 = SIZE*SIZE
    const ALIGN = 5
    const MAX_SCORE = 10
    const NUM_PLAYERS = 2

    const CUCB = 0.25 # Best UCB on 1000-MCTS matches. +2ELO vs 0.2 and +23ELO vs 0.3
    const HALFLIFE = 20 # In training, in each game temperature decreases at this rate : it halves every HALFLIFE moves

    const NN_TYPE = "Conv"
    const NUMBER_OUTPUTS = 82
    const NUM_FILTERS = 128
    const NUM_LAYERS = 10
    const V_HEAD = 8
    const PI_HEAD = 16

    #const PARAMS_MCTS_ELO = Dict([50=>442, 400=>1000, 800=>1161, 1600=>1313, 6400=>1495, 12800=>1555])  # for 664
    #const PARAMS_MCTS_ELO = Dict([50=>536, 400=>1000, 800=>1140, 1600=>1353, 6400=>1712])  # for 995
#     const PARAMS_MCTS_ELO = Dict([400=>1000, 800=>1140, 1600=>1353, 6400=>1712, 12800=>1885])  # for 995
#     const PARAMS_MCTS_ELO = Dict([12800=>1885, 25000=>2024])  # for 995



elseif GAME == "Connect4"
    const WIDTH = 7     #COLS
    const LENGTH = 6    #ROWS
    const SHAPE_INPUT = (LENGTH, WIDTH, 2)
    const SHAPE_BOARD = (LENGTH, WIDTH)
    const SHAPE_OUTPUT = (1, WIDTH, 1)
    const NUM_CELLS = WIDTH*LENGTH
    const NUMBER_ACTIONS = WIDTH

    const DIRECTORY = "Connect4_check/"

    const CUCB = 0.25
    const HALFLIFE = 15
    const NUM_PLAYERS = 2

    const NN_TYPE = "Conv"
    const NUM_FILTERS = 128
    const NUM_LAYERS = 10
    const V_HEAD = 8
    const PI_HEAD = 16

    const PARAMS_MCTS_ELO = Dict([50=>650, 400=>1000, 800=>1077, 1600=>1186, 6400=>1275, 12800=>1367]) # for 7x6


elseif GAME == "Kalah"
    const INIT_STONES = 4   # 4-kalah or 5-kalah or 6-kalah
    const SHAPE_INPUT = (15, 1, 1)
    const SHAPE_BOARD = (6, 2)
    const SHAPE_OUTPUT = (7, 1)
    const NUM_CELLS = 12
    const NUMBER_ACTIONS = 7
    const NUM_PLAYERS = 2

    const DIRECTORY = "Kalah/"
    const CUCB = 0.25
    const HALFLIFE = 15

    const NN_TYPE = "Dense"
    const NUM_FILTERS = 256
    const NUM_LAYERS = 7

#     const PARAMS_MCTS_ELO = Dict([50=>637, 400=>1000, 800=>1093, 1600=>1170, 6400=>1365, 12800=>1390])   #Pour KalahX without exchange
    const PARAMS_MCTS_ELO = Dict([50=>668, 400=>1000, 800=>1069, 1600=>1150, 6400=>1307, 12800=>1325])   #Pour KalahX with exchange

    #NECESSARY FOR THE OTHER GAMES
    const WIDTH = 1
    const LENGTH = 1

elseif GAME == "Santorini"
    const DIRECTORY = "Santorini/"

    const CUCB = 0.25   #Best UCB on 1000 & 5000 -MCTS matches. +18ELO vs 0.2 and +12ELO vs 0.3
    const HALFLIFE = 15

    const PARAMS_MCTS_ELO = Dict([200=>925, 400=>1138, 6400=>1640, 25600=>1879])

elseif GAME == "Boop"
    const DIRECTORY = "Boop/"

    const CUCB = 0.25   #Best UCB on 1000 & 5000 -MCTS matches. +18ELO vs 0.2 and +12ELO vs 0.
    const HALFLIFE = 30

    const PARAMS_MCTS_ELO = Dict([100=>570, 400=>1000, 1600 => 1629, 6400 => 2112])


elseif GAME == "Jaipur"
    const SHAPE_INPUT = (49, 1, 1)
    const SHAPE_BOARD = (5,1)
    const SHAPE_OUTPUT = (15, 1)
    const NUM_CELLS = 12
    const NUMBER_ACTIONS = 15

    const DIRECTORY = "Jaipur/"
    const CUCB = 0.25
    const HALFLIFE = 15
    const NUM_PLAYERS = 2

    const NN_TYPE = "Dense"
    const NUM_FILTERS = 256
    const NUM_LAYERS = 6

    const UNIVERSES = 4

#     const PARAMS_MCTS_ELO = Dict([1=>0, 10=>615, 40=>621, 160=>763])   #avec UNIVERSES = 10
#     const PARAMS_MCTS_ELO = Dict([1=>0, 50=>510, 200=>590, 800=>660])   #avec UNIVERSES = 2
    const PARAMS_MCTS_ELO = Dict([1=>0, 25=>710, 100=>730, 400=>819])   #avec UNIVERSES = 4   # MCTS is per universe !


    #NECESSARY FOR THE OTHER GAMES
    const WIDTH = 1
    const LENGTH = 1

elseif GAME == "Azul"
    const DIRECTORY = "Azul/"

    const CUCB = 0.25   #Best UCB on 1000 & 5000 -MCTS matches. +18ELO vs 0.2 and +12ELO vs 0.
    const HALFLIFE = 20.0

    const PARAMS_MCTS_ELO = Dict([100=>715, 400=>885, 1600 => 1054, 6400 => 1227])

elseif GAME == "Resolve"
    const DIRECTORY = "Resolve/"

    const CUCB = 1.5   #Best UCB on 1000 & 5000 -MCTS matches. +18ELO vs 0.2 and +12ELO vs 0.
    const HALFLIFE = 7.0

    const PARAMS_MCTS_ELO = Dict([100=>715, 400=>885, 1600 => 1054, 6400 => 1227])
else
    error("Erreur nom GAME dans params.jl : "*GAME)
end

const PARALLEL_RANDOM_NODES = 8

const GAME_SAVE_KEY = GAME
const DO_FORCED_PLAYOUTS = true
const kFP = 2.0
const FPU::Float32 = 0.2


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



# #MIDDLE b : neural network is good, and can still be better with a bigger batchsize.
#=
const LEARNING_PARAMS = Dict(
    "SIZE_REPLAYBUFFER" => 200000,
    "NGAMES" => 96,
    "lr_min" => 5e-5,
    "lr_max" => 5e-3,
    "MCTS_ITER" => 800,
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
=#

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


const BASE_HALFLIFE = exp(-1.0/HALFLIFE)


# OTHER CONSTANTS
const INF = 1000000.0
const TERMINAL_WIN = 3
const TERMINAL_DRAW = 2
const TERMINAL_LOSS = 1
const NOT_TERMINAL = 0
