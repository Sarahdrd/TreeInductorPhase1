using JSON, LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(ARGS[1], "QuantBnB-2D.jl"))
function main()
    d=JSON.parsefile(ARGS[2]);X=reduce(vcat, [permutedims(Float64.(r)) for r in d["X_train"]]);Z=reduce(vcat,[permutedims(Float64.(r)) for r in d["X_valid"]]);labels=Int.(d["y_train"]);k=d["classes"]
    Y=zeros(length(labels),k)
    for i in eachindex(labels);Y[i,labels[i]+1]=1;end
    started=time();g,gt=greedy_tree(X,Y,2,"C")
    ub,tree,remaining=QuantBnB_2D(X,Y,3,g*(1+1e-6),2,0.2,nothing,"C",false)
    elapsed=time()-started
    feasible=!(tree isa Number)
    out=Dict("feasible"=>feasible,"optimal"=>false,"certification"=>"not_exposed_by_original_API","solver_seconds"=>elapsed,"objective"=>ub,"remaining_rows"=>size(remaining,1),"greedy_upper_bound"=>g)
    if feasible
        train_scores=tree_eval(tree,X,2,k);valid_scores=tree_eval(tree,Z,2,k)
        out["train_predictions"]=[argmax(train_scores[i,:])-1 for i in axes(train_scores,1)];out["valid_predictions"]=[argmax(valid_scores[i,:])-1 for i in axes(valid_scores,1)]
        function shape(t)
            if length(t)!=4 || typeof(t[3])==typeof(1.0);return (0,0);end
            a,b=shape(t[3]),shape(t[4]);return (1+max(a[1],b[1]),1+a[2]+b[2])
        end
        depth,nodes=shape(tree);out["depth"]=depth;out["nodes"]=nodes;out["tree"]=tree
    end
    open(ARGS[3],"w") do f;JSON.print(f,out);end
end
main()
