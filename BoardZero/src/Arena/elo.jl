"""
    elo.jl

    every functions related to ELO computation
        * winrate(ELOa vs ELOb)
        * ELO(winrate of a vs b)
        * all ELOS(winrate of all a vs all b)
        * all ELOS(winrate of all a vs all b, some are restrained)
"""

module ELO

using PrettyTables

function winrate(elo_me, elo_he) :: Float64
    return 1.0 / (1+10.0 ^ ((elo_he - elo_me)/400.0))
end

function deltaELO(winrate)
    if winrate == 1.0
        return 5000
    elseif winrate == 0.0
        return -500
    else
        return 400*log10( winrate / (1-winrate))
    end
end

function relative_elos(results::Dict)
    agents = unique([[pair[1] for pair in keys(results)]..., [pair[2] for pair in keys(results)]...])
    ELO = Dict([x=>0 for x in agents])

    println("agents : ", agents, "ELO: ", ELO, " results: ", results)

    # compute loss function : cross entropy
    g(result, eloa, elob) = - result * log(winrate(eloa, elob))
    loss(elos) = sum( [g(results[pair], elos[pair[1]], elos[pair[2]]) for pair in keys(results)] ) + 0.5 * 1e-4 * sum([ELO[x]^2 for x in agents])
    grad(elos) = Dict( [ x=>
                    sum(Float64[winrate(elos[pair[1]], elos[pair[2]]) - results[pair] for pair in keys(results) if pair[1] == x]) -
                    sum(Float64[winrate(elos[pair[1]], elos[pair[2]]) - results[pair] for pair in keys(results) if pair[2] == x]) +
                    1e-4*ELO[x]
                for x in agents ] )

    norm_grad::Float64 = 1.0
    iter = 0
    step = 128
    while norm_grad > 1e-5 && iter < 20
        gradient = grad(ELO)
        norm_grad = maximum([abs(gradient[x]) for x in keys(gradient)])
        while loss( Dict([x=>ELO[x]-step*gradient[x] for x in keys(ELO)])  ) > loss(ELO)
            step /= 2
        end
        while loss( Dict([x=>ELO[x]-step*gradient[x] for x in keys(ELO)])  ) < loss(ELO)
            ELO = Dict([x=>ELO[x]-step*gradient[x] for x in keys(ELO)])
        end
        iter += 1
    end

    return ELO
end

function printing(results, ELOs)
    agents = unique([[pair[1] for pair in keys(results)]..., [pair[2] for pair in keys(results)]...])
    display(pretty_table(
        hcat([get(results, (x,y), -1) for x in agents, y in agents], reshape([ELOs[x] for x in agents], :, 1)) ; header=vcat(agents, "ELO"), row_labels=agents)
    )
end

function printing(results)
    ELOs = relative_elos(results)
    printing(results, ELOs)
end
    #=


function win_exp(elo_me, elo_he) :: Float64
    return 1 / (1+10.0 ^ ((elo_he - elo_me)/400.0))
end

function elo_util(elos, cur, target)
    # find elo s.t. elos[cur] match the other elos in the sense that : sum_x winrate(cur, x) = target
    elom, eloM = -10000, 10000
    while eloM - elom > 1
        elomid = (elom + eloM) / 2
        y = sum([win_exp(elomid, elo[2]) for elo in elos]) - win_exp(elomid, elos[cur])
        if y > target
            eloM = elomid
        else
            elom = elomid
        end
    end
    return (elom + eloM) / 2
end

function get_elo(winrate::Float64)
    if winrate == 1.0
        return 800
    elseif winrate == 0.0
        return -800
    else
        return 400*log10( winrate / (1-winrate))
    end
end=#
#=
function find_elos(scores, iterations, reference::Int)  # reference = 1000 ELO
    # all agents have played against themselves
    # agent number -1 is the one at 1000 ELO
    # Find elo_0 .. elo_(n-1) maximizing :
    #     prod_(i,j) P(win_ij = win_ijobs)
    # so maximizing the log. Moreover, P(win_ij = win_ijobs) follows a binomial distribution
    # so maximizing sum_(i,j) win_ijobs * log(win_ij) + (1-win_ijobs) log(1-win_ij)
    # so (Euler condition) for all i : sum_j win_ij = sum_j win_ijobs
    elos = Dict([x=>1000f0 for x in iterations])
    targets = Dict([x=>0.0f0 for x in iterations])
    for (key, score) in scores
        targets[key[1]] += score
    end

    for i in 1:200
        iter = rand(iterations)
        if iter == reference
            continue
        end
        elos[iter] = elo_util(elos, iter, targets[iter])
    end

    # print everything
    println(elos)
    println(iterations)
    display(pretty_table( [scores[(x,y)] for x in iterations, y in iterations] ; header=iterations, row_labels=iterations ))
    display(pretty_table( elos ))

end

function set_elo!(ELO, search_elo, results)
        mean() = sum([elo for (elo,s) in zip(ELO, search_elo) if !s]) / sum( .! search_elo)
        mean_prev = mean()
        g(result, eloa, elob) = - result * log(win_exp(eloa, elob))
        f(elos...) = sum( [g(results[pair], elos[pair[1]], elos[pair[2]]) for pair in keys(results)] )
#         gf(elos...) = Float64[!search_elo[i] ? Float64(0) :
#                     sum(Float64[win_exp(elos[pair[1]], elos[pair[2]]) - results[pair] for pair in keys(results) if pair[1] == i])
#                     for i in 1:length(list_ids)]
        gf(elos...) = Float64[ sum(Float64[win_exp(elos[pair[1]], elos[pair[2]]) - results[pair] for pair in keys(results) if pair[1] == i]) for i in 1:length(list_ids) ]
        norm_grad::Float64 = 1.0
        while norm_grad > 1e-5
            println(ELO)
            gradient = gf(ELO...)
            norm_grad = maximum(abs.(gradient))
            while f((ELO-gradient)...) < f(ELO...)
                ELO .= ELO - gradient
            end
        end
        mean_after = mean()
        ELO .= ELO .- mean_after .+ mean_prev
end=#

end
