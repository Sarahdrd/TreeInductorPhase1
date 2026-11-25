"""Publication figures from dataset summaries, with explicit F/H references."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .infrastructure import ROOT
import json

NAMES = {'murtree':'MurTree','quant_bnb':'Quant-BnB',
         'cart_depth2_murtree':'CART (MurTree features)',
         'cart_depth2_quant_bnb':'CART (Quant-BnB features)'}
COLORS = {'easy':'#0072B2','intermediate':'#009E73','hard':'#D55E00'}


def make_figures(tables, output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    metadata = json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names = {r['dataset']:r['canonical_name'] for r in metadata['datasets']}
    def save(fig,name):
        for ext in ['png','pdf']:
            path = output/(name+'.'+ext)
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    data = tables['dataset_performance']
    models = list(NAMES)
    fig,axes = plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    for ax,reference in zip(axes,['F','H']):
        for j,model in enumerate(models):
            subset = data[data.model.eq(model)]
            for i,stratum in enumerate(COLORS):
                vals = subset.loc[subset['stratum_'+reference].eq(stratum),'mean_macro_f1'].to_numpy()
                x = i+(j-1.5)*.18
                ax.scatter(x+np.linspace(-.035,.035,len(vals)),vals,s=22,alpha=.7,
                           color=plt.get_cmap('tab10')(j),label=NAMES[model] if i==0 else None)
                ax.plot([x-.06,x+.06],[vals.mean()]*2,color='black',lw=2)
        ax.set(xticks=range(3),xticklabels=['Easy (6)','Intermediate (6)','Hard (5)'],
               ylim=(0,1.03),title=reference+(' — primary five-seed strata' if reference=='F' else ' — historical sensitivity strata'),
               ylabel='Dataset mean validation macro-F1')
    axes[1].legend(loc='lower left',fontsize=8)
    fig.supxlabel('Points: dataset means over 75 folds; black segments: equal-weight stratum means',fontsize=9)
    fig.tight_layout();save(fig,'performance_by_stratum')

    d = tables['dataset_paired_deltas'].query('reference == "representation_matched_depth2_CART"')
    labels = tables['stratum_labels'].sort_values('rank_F');order = labels.dataset.tolist()
    fig,axes = plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,model in zip(axes,['murtree','quant_bnb']):
        g = d[d.model.eq(model)].set_index('dataset').loc[order]
        for i,(_,r) in enumerate(g.iterrows()):
            ax.hlines(i,0,r.mean_delta,color=COLORS[r.stratum_F],lw=2)
            ax.scatter(r.mean_delta,i,color=COLORS[r.stratum_F],s=35)
        ax.axvline(0,color='.3',lw=.8)
        ax.set(title=NAMES[model],xlabel='Mean paired Δ macro-F1 (solver − matched CART)',
               yticks=range(17),yticklabels=[names[k] for k in order])
        ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    fig.supxlabel('Order and colors use F: blue Easy, green Intermediate, orange Hard.\nH exchanges Sonar and Parkinsons; the dataset differences themselves are unchanged.',fontsize=9)
    fig.tight_layout();save(fig,'dataset_matched_differences')

    fig,ax = plt.subplots(figsize=(10,5.8))
    for j,model in enumerate(models):
        g=data[data.model.eq(model)].sort_values('rank_F')
        ax.scatter(g.mean_wall_seconds,j+np.linspace(-.18,.18,len(g)),
                   c=[COLORS[s] for s in g.stratum_F],s=30,alpha=.8)
    ax.set(xscale='log',yticks=range(4),yticklabels=[NAMES[m] for m in models],
           xlabel='Dataset mean process wall time (seconds; logarithmic scale)',
           title='Observed computational cost under the depth-two protocol')
    ax.grid(axis='x',alpha=.2)
    fig.supxlabel('One point per dataset; F colors as above. Includes startup/JIT, excludes preprocessing.\nNot a pure optimizer-speed comparison. External limit: 120 s per fit; job memory: 4 GiB.',fontsize=9)
    fig.tight_layout();save(fig,'runtime_by_method')

    status=tables['solver_status'].set_index('model').loc[models]
    fig,ax=plt.subplots(figsize=(10,4.7))
    bars=ax.barh(range(4),status.fits,color=['#0072B2','#E69F00','#999999','#999999'])
    for bar,(_,r) in zip(bars,status.iterrows()):
        ax.text(20,bar.get_y()+bar.get_height()/2,
                f'{r.termination_status.replace("_"," ")} ({r.fits})',va='center',fontsize=9)
    ax.set(yticks=range(4),yticklabels=[NAMES[m] for m in models],xlim=(0,1400),
           xlabel='Number of completed evaluations',title='Complete coverage: 5,100 / 5,100; timeouts and failures: 0')
    fig.supxlabel('MurTree certification is restricted to the supplied binary representation and depth/node constraints.\nQuant-BnB returns feasible, uncertified solutions; CART is heuristic.',fontsize=9)
    fig.tight_layout();save(fig,'solver_status')
