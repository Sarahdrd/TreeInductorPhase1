"""Publication figures for descriptive label stability, using computed tables."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from .infrastructure import ROOT,config


def make_figures(tables,output,*,resume_incomplete=False):
    manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names={d['dataset']:d['canonical_name'] for d in manifest['datasets']}
    colors=['#0072B2','#009E73','#D55E00'];labels=['easy','intermediate','hard']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        if resume_incomplete and all((output/(name+'.'+ext)).exists() for ext in ['png','pdf']):
            plt.close(fig)
            return
        for extension in ['png','pdf']:
            path=output/(name+'.'+extension)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    d=tables['canonical_ranking_with_stability'].sort_values('rank')
    y=np.arange(17);fig,ax=plt.subplots(figsize=(10,8))
    for offset,column,label,color,marker in zip([-.2,0,.2],['seed_stability','weight_stability','combined_stability'],
            ['Seed (5)','Weight (5)','Seed × Weight (25)'],colors,['o','s','D']):
        ax.scatter(d[column],y+offset,label=label,color=color,marker=marker,s=38)
    ax.set(yticks=y,yticklabels=[names[x] for x in d.dataset],xlim=(-.04,1.04),xticks=np.arange(0,1.01,.2),
        xlabel='Fraction retaining the five-seed canonical label',title='Label stability under the prespecified sensitivity grid')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=3,frameon=False)
    save(fig,'label_stability')
    grid=tables['scenario_rankings'];combined=grid.loc[grid.family.eq('combined')]
    scenarios=list(combined.scenario.drop_duplicates())
    matrix=combined.pivot(index='dataset',columns='scenario',values='stratum').loc[d.dataset,scenarios]
    encode={label:i for i,label in enumerate(labels)}
    values=np.column_stack([d.stratum.map(encode),matrix.map(encode.__getitem__).to_numpy()])
    fig,ax=plt.subplots(figsize=(15,8))
    ax.imshow(values,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),aspect='auto',interpolation='nearest')
    ticks=['Reference']+[f'{a:.2f}' for _ in config()['cv']['seeds'] for a in [.51,.60,.70,.80,.90]]
    ax.set(xticks=range(26),xticklabels=ticks,yticks=range(17),yticklabels=[names[x] for x in d.dataset],xlabel='Alpha within each seed (reference = five-seed aggregate, α = 0.70)',title='Stratum assignments across 25 related scenarios')
    ax.tick_params(axis='x',labelrotation=90)
    for j,seed in enumerate(config()['cv']['seeds']):
        ax.text(3+5*j,-1.05,f'Seed {seed}',ha='center',fontsize=10)
        ax.axvline(.5+5*j,color='white',lw=2.5)
    for edge in [5.5,11.5]: ax.axhline(edge,color='white',lw=1.5)
    ax.legend(handles=[Patch(color=c,label=l.title()) for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False)
    save(fig,'scenario_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,edge,title in zip(axes,[6,12],['Easy–Intermediate boundary','Intermediate–Hard boundary']):
        side='easy_intermediate_crossed' if edge==6 else 'intermediate_hard_crossed'
        crossing=set(grid.loc[grid[side],'dataset'])
        selected=set(d.loc[d['rank'].between(edge-2,edge+3),'dataset'])|crossing
        focus=d.loc[d.dataset.isin(selected)].sort_values('rank')
        for i,(_,r) in enumerate(focus.iterrows()):
            g=combined.loc[combined.dataset.eq(r.dataset)]
            ax.plot([g.difficulty_score.min(),g.difficulty_score.max()],[i,i],color='.65',lw=2)
            for alpha,color in zip([.51,.60,.70,.80,.90],['#440154','#3B528B','#21918C','#5EC962','#D8B500']):
                selected_scores=g.loc[g.alpha.eq(alpha),'difficulty_score']
                ax.scatter(selected_scores,np.full(5,i),color=color,s=15,alpha=.7)
            ax.scatter(r.difficulty_score,i,marker='D',s=55,facecolor='white',edgecolor='black',zorder=5)
        ax.set(yticks=range(len(focus)),yticklabels=[names[x] for x in focus.dataset],xlabel='Difficulty score',title=title)
        ax.invert_yaxis()
        for tick in ax.get_yticklabels():
            if tick.get_text() in ['Sonar','Parkinsons']: tick.set_color('#D55E00');tick.set_fontweight('bold')
        gap=tables['canonical_boundary_gaps'].loc[tables['canonical_boundary_gaps'].lower_rank.eq(edge)].iloc[0]
        ax.axvline(gap.descriptive_midpoint,color='.4',ls=':',lw=1)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='D',color='none',markeredgecolor='black',markerfacecolor='white',label='Canonical score')]
    handles += [Line2D([0],[0],marker='o',color=c,lw=0,label=f'α = {a:.2f}') for a,c in zip([.51,.60,.70,.80,.90],['#440154','#3B528B','#21918C','#5EC962','#D8B500'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.11),ncol=6,frameon=False)
    fig.text(.5,-.01,'Dots: 25 seed × weight scores. Dotted lines: canonical score midpoints, descriptive only; labels use ranks.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'boundary_sensitivity')
    fig,ax=plt.subplots(figsize=(8,6));agreement=tables['rank_agreement_summary']
    for family,label,color,marker in [('combined','Seed × Weight',colors[0],'o'),('aggregate_weight','Aggregate Weight',colors[2],'s'),('seed','Seed at α = 0.70',colors[1],'D')]:
        g=agreement.loc[agreement.family.eq(family)]
        ax.scatter(g.spearman_rank,g.changed_strata,color=color,marker=marker,s=65,label=label,alpha=.7,edgecolors='white',linewidths=.5)
    ax.set(xlabel='Spearman rank agreement with five-seed canonical ranking',ylabel='Datasets with changed stratum (out of 17)',
        title='High rank agreement can coexist with label changes',yticks=range(int(agreement.changed_strata.max())+1))
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False);ax.grid(alpha=.15)
    fig.text(.5,-.01,'Overlapping points are retained; seed-at-0.70 scenarios are also members of the combined grid.',ha='center',fontsize=9)
    save(fig,'rank_vs_label_agreement')
