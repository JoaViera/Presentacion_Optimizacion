using StatsBase
using LinearAlgebra
using Statistics

# Función para calcular el SSE
function calculate_sse(data, centroids, clusters)
    sse = 0.0
    for i in 1:size(data, 1)
        # Obtener el centroide correspondiente al clúster asignado
        center = centroids[clusters[i],:]
        # Sumar la distancia al cuadrado entre el punto y su centroide
        sse += sum((data[i, :] .- center).^2)
    end
    return sse
end

# inicializacion aleatoria de los clusters
# K = cantidad de clusters, n = tamaño de la poblacion
function init_poblation(K,l)
    p = Int(floor(l/K))
    W = zeros(Int64,l)
    for k in 1:K # cada cluster empieza con la misma cantidad minima de individuos
        posiciones_disponibles = [i for i in 1:l if W[i] == 0]
        cluster_k = sample(posiciones_disponibles, p; replace = false)
        for j in cluster_k
            W[j] = k
        end
    end
    for i in 1:l # los individuos que sobran los asigno de forma aleatoria a los clusters
        if W[i] == 0
            W[i] = rand(1:K)
        end
    end
    return W
end

function centroides(data,w,k)
    n = size(data,1) # cantidad de datos
    l = size(data,2) # tamaño de cada dato
    # matriz de k x l donde cada fila tiene el centroide del cluster k
    centroid = zeros(k,l)
    cluster_size = zeros(k)
    for i in 1:n
        j = w[i] # el i-esimo dato pertenece al cluster j
        cluster_size[j] += 1 # agrego un elemento al cluster
        centroid[j, :] += data[i,:] # sumo el dato al centroide
    end
    for h in 1:k
        # promedio por la cantidad de datos en el cluster
        centroid[h,:] *= 1/cluster_size[h]
    end
    return centroid
end

function S_k(data, centroids, w, k)
    n = size(data,1) # cantidad de datos 
    res = 0
    for i in 1:n
        if w[i] == k
            res += norm(data[i,:] - centroids[k,:])^2
        end
    end
    return res
end

function S(data, centroids, w)
    n = size(data,1) # cantidad de datos 
    res = 0
    for i in 1:n 
        j = w[i]
        res += norm(data[i,:] - centroids[j,:])^2
    end
    return res
end

function Fitness(S_pob; c = 2) # le paso las S(w) de cada w en la poblacion,  c cosntante en [1,3]
    prom = -1*mean(S_pob)
    dev = std(S_pob)
    n = length(S_pob)
    res = zeros(n)
    for i in 1:n
        temp = -1*S_pob[i] - (prom - c*dev)
        if temp>0
            res[i] = temp
        end
    end
    return res
end

# necesito seleccionar una cantidad de individuos igual al tamaño de la poblacion
# realizamos fitness_proportionate_selection
function selection(poblation, Fitness_pob)
    l = length(Fitness_pob)
    if Fitness_pob == zeros(l)
        Fitness_pob = ones(l)
    end
    for i in 2:l # convierto f en un vector distribucional
        Fitness_pob[i] += Fitness_pob[i-1]
    end

    # realizamos stochastic_universal_sampling
    value = rand() * Fitness_pob[l]/l
    i = 1
    select_P = [Vector{Int64}() for _ in 1:l]
    for j in 1:l
        while Fitness_pob[i] < value
            i += 1
        end
        select_P[j] = Vector{Int64}(poblation[i])
        value += Fitness_pob[l]/l
    end
    
    return select_P
end

function mutation(data, w, K, p; c_m = 2)
    n = size(data,1)
    for i in 1:n
        if rand() < p
            d = zeros(K)
            # calcular centroides segun w
            centroids = centroides(data,w,K)
            # calculo las distancias a los centroides
            for j in 1:K
                d[j] = norm(data[i,:] - centroids[j,:])^2
            end
            if d[w[i]] > 0
                d_max = maximum(d)
                prob = zeros(K)
                for j in 1:K
                    prob[j] = (c_m*d_max - d[j])
                end
                w[i] = sample(1:K, Weights(prob))
            end
        end
    end
    return w
end

function K_means_operator(data, w, K)
    # realizo una iteracion de K_means
    centroids = centroides(data,w,K)
    n = size(data,1)
    for i in 1:n
        d = zeros(K)
        for j in 1:K
            d[j] = norm(data[i,:] - centroids[j,:])^2
        end
        w[i] = argmin(d)
    end

    clusters_vacios = setdiff(1:K, w)
    S_clusters = zeros(K)

    for vacio in clusters_vacios

        # actualizo los centroides
        centroids = centroides(data,w,K)
        
        # calculo el S_k(w) de cada cluster
        for j in 1:K
            S_clusters[j] = S_k(data, centroids, w, j)
        end

        # agarro el cluster con mayor S_k
        cluster_max = argmax(S_clusters)
        # calculo las distancias de sus puntos al centroide
        centroide_max = centroids[cluster_max,:]

        d_max = [w[i] == cluster_max ? norm(data[i,:] - centroide_max) : 0 for i in 1:n]
        
        indice_lejano = argmax(d_max)

        w[indice_lejano] = vacio
    end
    return w
end

# la data viene en forma matricial tal que data[i,j] es el x_ij descrito arriba
# o sea data[i,j] es el feature j-esimo del i-esimo individuo
# K cantidad de cluster a formar, popsize la cantidad de individuos en cada generacion
# max_gen cantidad de generaciones y p la probabilidad de mutar
function GeneticKMeans(data, K, popsize, max_gen, p; poblacion_inicial = nothing)
    # cantidad de genes en cada individuo, o sea cantidad de datos ejemplo
    l = size(data,1)
    poblation = Vector{Vector{Int64}}()
    if isnothing(poblacion_inicial)
        # inicializo popsize individuos de forma aleatoria
        for i in 1:popsize
            push!(poblation,init_poblation(K,l))
        end
    else
        for i in 1:popsize
            push!(poblation,poblacion_inicial)
        end
    end

    # vector del mejor de cada generacion
    bests = Vector{Vector{Int64}}([])

    # calculo los valores de fitness para la primera iteracion
    S_pob = zeros(popsize)
    for i in 1:popsize
        centroids = centroides(data,poblation[i],K)
        S_pob[i] = S(data, centroids, poblation[i])
    end
    Fitness_pob = Fitness(S_pob)

    for _ in 1:max_gen
        # selecciono los que dejaran descendencia
        Q = selection(poblation, Fitness_pob)

        # muto los individuos
        for i in 1:popsize
            poblation[i] = mutation(data, Q[i], K, p)
        end

        # realizo K-means
        for i in 1:popsize
            poblation[i] = K_means_operator(data, poblation[i], K)
        end
        
        # calculo el valor de fitness para cada individuo de la poblacion
        for i in 1:popsize
            centroids = centroides(data,poblation[i],K)
            S_pob[i] = S(data, centroids, poblation[i])
        end
        Fitness_pob = Fitness(S_pob)

        # actualizo best
        indice_best = argmax(Fitness_pob)
        push!(bests,poblation[indice_best])
    end
    return bests
end