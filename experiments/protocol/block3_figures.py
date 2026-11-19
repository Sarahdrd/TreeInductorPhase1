"""Publication figures for Block 3; reads computed tables only."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .infrastructure import ROOT


def make_figures(tables, output):
    manifest = json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names = {d['dataset']:d['canonical_name'] for d in manifest['datasets']}
    colors = dict(easy='#0072B2', intermediate='#009E73', hard='#D55E00')
    handles = [Line2D([0],[0], color=color, marker='o', label=label.title()) for label,color in colors.items()]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig, name):
        fig.savefig(output/(name+'.png'), dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(output/(name+'.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    primary = tables['five_seed_primary'].set_index('dataset')
    new = tables['unified_seed42'].set_index('dataset')
    fig, ax = plt.subplots(figsize=(11,8.5))
    for dataset, row in primary.iterrows():
        old = new.loc[dataset]
        ax.plot([0,1],[old['rank'],row['rank']],'-o',color=colors[row.stratum],lw=1.5,ms=4,alpha=.85)
        ax.text(-.035,old['rank'],names[dataset],ha='right',va='center',fontsize=9)
        ax.text(1.035,row['rank'],names[dataset],ha='left',va='center',fontsize=9)
    for boundary in [6.5,12.5]:
        ax.axhline(boundary,color='.65',lw=.7,ls='--',zorder=0)
    ax.set(xlim=(-.7,1.7),ylim=(17.7,.3),xticks=[0,1],xticklabels=['Unified seed 42','Five-seed primary'],
           ylabel='Difficulty rank (1 = easiest)',yticks=[1,6,7,12,13,17],
           title='Canonical difficulty ranks: single seed versus five seeds')
    ax.spines['bottom'].set_visible(False)
    ax.legend(handles=handles, title='Five-seed stratum', loc='upper center', bbox_to_anchor=(.5,-.07), ncol=3, frameon=False)
    save(fig,'rank_comparison')

    comparison = tables['ranking_comparison'].sort_values('five_seed_rank')
    fig, ax = plt.subplots(figsize=(9,7))
    positions = np.arange(len(comparison))
    ax.barh(positions,comparison.rank_displacement,color=[colors[x] for x in comparison.five_seed_stratum],height=.68)
    ax.set(yticks=positions,yticklabels=[names[x] for x in comparison.dataset],
           xlabel='Five-seed rank − unified seed-42 rank (positive = harder)',
           title='Rank displacement under five-seed aggregation')
    ax.invert_yaxis(); ax.axvline(0,color='.3',lw=.8)
    limit=max(1,int(comparison.rank_displacement.abs().max()))
    ax.set_xlim(-limit-.7,limit+.7); ax.set_xticks(range(-limit,limit+1))
    for y,delta in zip(positions,comparison.rank_displacement):
        ax.text(delta+(.1 if delta>=0 else -.1),y,f'{delta:+d}',va='center',ha='left' if delta>=0 else 'right')
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,-.11), ncol=3, frameon=False)
    save(fig,'rank_displacement')

    selected=set(primary.loc[primary['rank'].between(10,15)].index)|{'sonar','parkinsons','haberman'}
    focus=primary.loc[list(selected)].sort_values('rank')
    fig,ax=plt.subplots(figsize=(9,5))
    for y,(dataset,row) in enumerate(focus.iterrows()):
        ax.plot([new.loc[dataset,'difficulty_score'],row.difficulty_score],[y,y],color='.7',lw=1)
        ax.scatter(new.loc[dataset,'difficulty_score'],y,marker='o',facecolors='white',edgecolors='#0072B2',s=55,
                   label='Unified seed 42' if y==0 else None)
        ax.scatter(row.difficulty_score,y,marker='s',color='#D55E00',s=42,label='Five-seed primary' if y==0 else None)
    midpoint=(primary.loc[primary['rank'].eq(12),'difficulty_score'].iloc[0]+primary.loc[primary['rank'].eq(13),'difficulty_score'].iloc[0])/2
    ax.axvline(midpoint,color='.45',ls=':',lw=1,label='Five-seed rank 12/13 score midpoint')
    ax.set(yticks=range(len(focus)),yticklabels=[f'{names[d]}  (rank {int(r)})' for d,r in zip(focus.index,focus['rank'])],
           xlabel='Predictive-difficulty score',title='Datasets near the Intermediate–Hard boundary')
    ax.invert_yaxis();ax.legend(loc='upper left',bbox_to_anchor=(0,-.16),frameon=False,fontsize=9)
    save(fig,'intermediate_hard_boundary')

    pooled=tables['five_seed_pooled_sd_sensitivity'].set_index('dataset').loc[primary.index]
    fig,ax=plt.subplots(figsize=(6.5,6))
    ax.plot([0,1],[0,1],color='.65',lw=1,ls='--',label='Equal score')
    ax.scatter(primary.difficulty_score,pooled.difficulty_score,color=[colors[s] for s in primary.stratum],s=36)
    for dataset in ['sonar','parkinsons','haberman']:
        offset={'sonar':(-48,20),'parkinsons':(14,-24),'haberman':(18,18)}[dataset]
        ax.annotate(names[dataset],(primary.loc[dataset,'difficulty_score'],pooled.loc[dataset,'difficulty_score']),
                    xytext=offset,textcoords='offset points',fontsize=9,
                    arrowprops=dict(arrowstyle='-',color='.4',lw=.7))
    ax.set(xlabel='Primary: mean within-seed SD',ylabel='Sensitivity: pooled 75-fold SD',
           title='Difficulty-score sensitivity to variability definition',xlim=(-.02,1.02),ylim=(-.02,1.02))
    ax.legend(handles=handles, title='Primary stratum', loc='upper left', frameon=False)
    ax.set_aspect('equal');save(fig,'variability_sensitivity')
