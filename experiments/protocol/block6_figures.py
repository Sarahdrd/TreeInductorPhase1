"""Scientific visualizations of saved paired tree effects."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from .infrastructure import ROOT

PANEL_NAMES={'leaf_2':'Minimum Leaf Size = 2','leaf_5':'Minimum Leaf Size = 5','ccp_001':'CCP Alpha = 0.001','ccp_01':'CCP Alpha = 0.01'}
COLORS={'easy':'#0072B2','intermediate':'#009E73','hard':'#D55E00'}


def make_figures(tables,output):
    manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names={d['dataset']:d['canonical_name'] for d in manifest['datasets']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:
            path=output/(name+'.'+ext)
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    d=tables['mixed_vs_dt_only'].sort_values('rank_reference')
    fig,ax=plt.subplots(figsize=(12,8.5))
    for _,r in d.iterrows():
        color='#D55E00' if r.stratum_changed else '#0072B2'
        ax.plot([0,1],[r.rank_reference,r.rank_alternative],'-o',color=color,ms=4,lw=1.5)
        ax.text(-.04,r.rank_reference,names[r.dataset],ha='right',va='center',fontsize=9)
        ax.text(1.04,r.rank_alternative,names[r.dataset],ha='left',va='center',fontsize=9)
    for edge in [6.5,12.5]:ax.axhline(edge,ls='--',color='.6',lw=.8)
    ax.set(xlim=(-.85,1.85),ylim=(17.7,.3),xticks=[0,1],xticklabels=['Mixed DT/LR canonical','DT-Only unpruned'],
        ylabel='Difficulty rank (1 = easiest)',title='Five-seed ranking comparison | Orange: changed stratum')
    save(fig,'mixed_vs_dt_only')
    agreements=tables['regularization_rank_agreement'];fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ref,color,offset in [('mixed','#0072B2',-.12),('dt_only','#D55E00',.12)]:
        g=agreements.loc[agreements.reference.eq(ref)].set_index('panel').loc[list(PANEL_NAMES)]
        for ax,metric in zip(axes,['spearman_rank','changed_strata']):
            ax.scatter(g[metric],np.arange(4)+offset,color=color,s=50,label='Mixed DT/LR' if ref=='mixed' else 'DT-Only')
            ax.set(yticks=range(4),yticklabels=list(PANEL_NAMES.values()))
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
    axes[0].set(xlabel='Spearman rank correlation',title='Rank agreement by reference')
    axes[1].set(xlabel='Changed stratum assignments (out of 17)',title='Label disagreement by reference',xticks=range(int(agreements.changed_strata.max())+1))
    axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
    fig.tight_layout();save(fig,'regularization_rank_agreement')
    effects=tables['dataset_regularization_effects'];order=d.dataset.tolist()
    matrix=effects.pivot(index='dataset',columns='panel',values='mean_delta').loc[order,list(PANEL_NAMES)]
    maximum=float(np.abs(matrix.to_numpy()).max())
    fig,ax=plt.subplots(figsize=(9,8.5));im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum),aspect='auto')
    ax.set(yticks=range(17),yticklabels=[names[x] for x in order],xticks=range(4),
        xticklabels=['Leaf Size 2','Leaf Size 5','CCP 0.001','CCP 0.01'],title='Mean paired macro-F1 change: regularized minus unpruned')
    for i in range(17):
        for j in range(4):
            value=matrix.iloc[i,j];ax.text(j,i,f'{value:+.3f}',ha='center',va='center',color='white' if abs(value)>maximum*.55 else 'black',fontsize=9)
    for edge in [5.5,11.5]:ax.axhline(edge,color='black',lw=1.5)
    fig.colorbar(im,ax=ax,label='Mean paired Δ macro-F1 (equal class-weight averaging)',pad=.03)
    save(fig,'dataset_regularization_effects')
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharey=True)
    for ax,(panel,title) in zip(axes.flat,PANEL_NAMES.items()):
        g=effects.loc[effects.panel.eq(panel)]
        for i,(label,color) in enumerate(COLORS.items()):
            v=g.loc[g.canonical_stratum.eq(label)].sort_values('canonical_rank')
            offsets=np.linspace(-.15,.15,len(v))
            ax.scatter(i+offsets,v.mean_delta,color=color,s=40,zorder=3)
            ax.plot([i-.2,i+.2],[v.mean_delta.median()]*2,color='black',lw=2)
            # Identify the strongest positive and negative effects in each panel.
        for index in [g.mean_delta.idxmin(),g.mean_delta.idxmax()]:
            r=g.loc[index];i=list(COLORS).index(r.canonical_stratum)
            ax.annotate(names[r.dataset],(i,r.mean_delta),xytext=(0,9 if r.mean_delta>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(0,color='.45',ls='--',lw=.8);ax.set(xticks=range(3),xticklabels=[x.title() for x in COLORS],title=title,ylabel='Dataset mean paired Δ macro-F1')
    fig.suptitle('Regularization effects within fixed mixed canonical strata',y=1.01)
    fig.text(.5,-.01,'Each point is one dataset; short black lines mark medians. No fold-level independence is assumed.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'stratum_regularization_effects')
    cw=tables['class_weight_comparison'];fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    extent=max(abs(cw[['mean_delta_unbalanced','mean_delta_balanced']].to_numpy()).max()*1.13,.01)
    for ax,(panel,title) in zip(axes.flat,PANEL_NAMES.items()):
        g=cw.loc[cw.panel.eq(panel)]
        for label,color in COLORS.items():
            v=g.loc[g.canonical_stratum.eq(label)]
            ax.scatter(v.mean_delta_unbalanced,v.mean_delta_balanced,color=color,s=42,label=label.title())
        ax.plot([-extent,extent],[-extent,extent],ls=':',color='.55',lw=1)
        ax.axhline(0,color='.6',lw=.7);ax.axvline(0,color='.6',lw=.7)
        ax.set(title=title,xlabel='Unbalanced tree: mean paired Δ macro-F1',ylabel='Balanced tree: mean paired Δ macro-F1',xlim=(-extent,extent),ylim=(-extent,extent))
    handles=[Line2D([0],[0],marker='o',lw=0,color=c,label=l.title()) for l,c in COLORS.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.05),ncol=3,frameon=False)
    fig.tight_layout();save(fig,'class_weight_sensitivity')
