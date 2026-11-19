"""Publication figures from saved Block 4 tables; no model operations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_figures(tables,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    d=tables['original_vs_corrected'].sort_values('original_rank')
    def save(fig,name):
        for extension in ['png','pdf']:
            path=output/(name+'.'+extension)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,8.5))
    for _,r in d.iterrows():
        color='#D55E00' if r.stratum_changed else '#0072B2'
        ax.plot([0,1],[r.original_rank,r.corrected_rank],'-o',color=color,lw=1.5,ms=4)
        ax.text(-.04,r.original_rank,r.dataset_name,ha='right',va='center',fontsize=9)
        ax.text(1.04,r.corrected_rank,r.dataset_name,ha='left',va='center',fontsize=9)
    for edge in [6.5,12.5]: ax.axhline(edge,color='.6',ls='--',lw=.8)
    ax.set(xlim=(-.85,1.85),ylim=(17.7,.3),xticks=[0,1],xticklabels=['Original five-seed score','Chance-corrected alternative'],
        ylabel='Difficulty rank (1 = easiest)',title='Rank comparison | Orange: changed stratum; blue: unchanged')
    save(fig,'rank_comparison')
    for columns,labels,title,xlabel,name in [
        (['original_difficulty','chance_corrected_difficulty'],['Original five-seed score','Chance-corrected alternative'],
         'Difficulty scores across the fixed benchmark','Difficulty score','score_comparison'),
        (['stratified_dummy_macro_f1','original_mean_macro_f1'],['Stratified dummy','Canonical DT/LR mean'],
         'Saved five-seed macro-F1 means','Macro-F1','dummy_vs_canonical')]:
        fig,ax=plt.subplots(figsize=(10,8));y=np.arange(len(d))
        for i,(_,r) in enumerate(d.iterrows()): ax.plot([r[columns[0]],r[columns[1]]],[i,i],color='.75',lw=1)
        for column,label,color,marker in zip(columns,labels,['#0072B2','#D55E00'],['o','s']):
            ax.scatter(d[column],y,label=label,color=color,marker=marker,s=35,zorder=3)
        ax.set(yticks=y,yticklabels=d.dataset_name,xlabel=xlabel,title=title,xlim=(-.02,1.02))
        ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    for ax,column,title,color in zip(axes,['original_difficulty','chance_corrected_difficulty'],
            ['Original five-seed score','Chance-corrected alternative'],['#0072B2','#D55E00']):
        ax.scatter(d.minimum_class_count,d[column],color=color,s=35)
        for dataset,offset in [('heart_disease',(32,6)),('wine_red',(32,-25))]:
            r=d.loc[d.dataset.eq(dataset)].iloc[0]
            ax.annotate(r.dataset_name,(r.minimum_class_count,r[column]),xytext=offset,textcoords='offset points',
                fontsize=9,arrowprops=dict(arrowstyle='-',color='.4',lw=.7))
        ax.set(xscale='log',xlabel='Minimum class count (log scale)',title=title,ylim=(-.04,1.06))
    axes[0].set_ylabel('Difficulty score')
    fig.suptitle('Exploratory dataset-level association (17 datasets)',y=1.02)
    save(fig,'class_structure')
