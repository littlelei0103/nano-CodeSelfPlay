"""Render release figures from bundled raw reports; no checkpoints or fallback scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from scripts.aggregate_all_reports import GROUPS, calc_stats

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'base': '#87929d', 'breaker_only_static': '#7856a3', 'pure_selfplay': '#168c82',
          'mixed_replay': '#3878bd', 'frozen_mutator_ablation': '#c87c37'}
LABELS = {'base':'Base', 'breaker_only_static':'Static GRPO', 'pure_selfplay':'Self-play',
          'mixed_replay':'Mixed replay', 'frozen_mutator_ablation':'Frozen Mutator'}


def prepare_plot_data(summary):
    if summary.get('schema_version') != 2:
        raise ValueError('Rebuild the validated summary before plotting')
    data = {'test': {}, 'trajectories': {}}
    base = summary['base_model']['test_benchmark']
    data['test']['base'] = {
        'test_greedy_pass1': [base['greedy_pass1']],
        'test_sample_pass4': list(base['sample_pass4_by_seed'].values()),
        'test_hit_rate': list(base['hit_density_by_seed'].values()),
    }
    for key in GROUPS:
        group = summary[key]
        runs = [group['runs'][str(seed)] if str(seed) in group['runs'] else group['runs'][seed]
                for seed in group['seeds']]
        if len(runs) != 3:
            raise ValueError(f'{key}: release figures require three completed seeds')
        data['test'][key] = {}
        for metric in ('test_greedy_pass1', 'test_sample_pass4', 'test_hit_rate'):
            vals = [r[metric] for r in runs]
            actual = calc_stats(vals)
            if not np.allclose(actual, group[metric]):
                raise ValueError(f'{key}/{metric}: aggregate and raw values disagree')
            data['test'][key][metric] = vals
        steps = [[r['step'] for r in run['trajectory']] for run in runs]
        if any(s != steps[0] for s in steps):
            raise ValueError('Validation trajectories are incomplete or use different schedules')
        data['trajectories'][key] = {'steps':steps[0]}
        for metric in ('val_greedy_pass1','pool_size'):
            vals = [[r[metric] for r in run['trajectory']] for run in runs]
            data['trajectories'][key][metric] = {'mean':np.mean(vals,axis=0).tolist(),
                                                'std':np.std(vals,axis=0,ddof=1).tolist()}
    return data


def style_axes(ax):
    ax.set_facecolor('#ffffff')
    ax.grid(axis='x', color='#e6eaf0', linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ('top','right','left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#d4dbe4')
    ax.tick_params(length=0, pad=7)


def save_figure(fig, output, name):
    for suffix in ('png','svg'):
        fig.savefig(output/f'{name}.{suffix}', dpi=180, facecolor='white', bbox_inches='tight',
                    **({'metadata': {'Date': None}} if suffix == 'svg' else {}))
        if suffix == 'svg':
            path = output/f'{name}.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def plot_test(data, output):
    order = ['base','breaker_only_static','pure_selfplay','mixed_replay','frozen_mutator_ablation']
    metrics = [('test_greedy_pass1','Greedy Pass@1'),('test_sample_pass4','Sample Pass@4'),('test_hit_rate','Candidate hit rate')]
    names = ['Base', 'Static\nGRPO', 'Self-play', 'Mixed\nreplay', 'Frozen\nMutator']
    fig, axes = plt.subplots(1,3,figsize=(13.8,5.6),sharey=True)
    fig.subplots_adjust(left=.055,right=.985,bottom=.25,top=.73,wspace=.15)
    fig.suptitle('Finding counterexamples',x=.055,y=.97,ha='left',fontsize=22,fontweight='bold',color='#1c2d42')
    fig.text(.055,.89,'Final test · 20 problems · checkpoints selected on validation',fontsize=11,color='#66758a')
    fig.text(.055,.835,'Three training seeds: 42 / 43 / 44',fontsize=10,color='#66758a')
    for ax,(metric,title) in zip(axes,metrics):
        style_axes(ax)
        ax.grid(axis='x',visible=False)
        ax.grid(axis='y',color='#e6eaf0',linewidth=.8)
        ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=18,color='#1c2d42')
        for x,key in enumerate(order):
            vals=np.array(data['test'][key][metric])*100
            mean,std=calc_stats(vals.tolist())
            ax.bar(x,mean,width=.58,color=COLORS[key],alpha=.8,zorder=3)
            if std is not None and std > 0:
                ax.errorbar(x,mean,yerr=std,fmt='none',ecolor='#34475f',
                            elinewidth=1.2,capsize=3,capthick=1.2,zorder=4)
            top=mean+(std or 0)
            ax.text(x,top+10,f'{mean:.1f}%',ha='center',va='bottom',
                    fontsize=10,fontweight='bold',color='#1c2d42')
            if std is not None:
                ax.text(x,top+3,f'±{std:.1f}',ha='center',va='bottom',
                        fontsize=8.5,color='#66758a')
        ax.set_xlim(-.65,len(order)-.35)
        ax.set_ylim(0,115)
        ax.set_xticks(range(len(order)),names,fontsize=9)
        ax.set_yticks([0,25,50,75,100])
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.tick_params(axis='x',pad=10)
    fig.text(.055,.10,'Bars: means  ·  Error bars and ± values: sample SD in percentage points, not confidence intervals.',fontsize=9.5,color='#52637a')
    fig.text(.055,.055,'Base sampling uses three decoding seeds of one model; Base Greedy is deterministic.',fontsize=9,color='#788598')
    fig.text(.055,.015,'Training configurations are not compute-matched. Individual seed results are included in the reports.',fontsize=9,color='#788598')
    save_figure(fig,output,'v2_regime_comparison')


def plot_training(data, summary, output):
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.7))
    fig.subplots_adjust(left=.075,right=.98,bottom=.25,top=.76,wspace=.22)
    fig.suptitle('Training traces · three seeds per configuration',x=.075,y=.97,ha='left',fontsize=18,fontweight='bold',color='#1c2d42')
    fig.text(.075,.86,'Lines show means; shaded regions show sample SD across training seeds.',fontsize=10.5,color='#66758a')
    for ax,metric,title in zip(axes,['val_greedy_pass1','pool_size'],['Validation Greedy Pass@1','Unique generated pairs in HardBuffer']):
        for key in GROUPS:
            if metric=='pool_size' and key=='breaker_only_static':continue
            trace=data['trajectories'][key];mean=np.array(trace[metric]['mean']);std=np.array(trace[metric]['std'])
            if metric=='val_greedy_pass1':mean,std=mean*100,std*100
            ax.plot(trace['steps'],mean,color=COLORS[key],lw=2,marker='o',ms=4,label=LABELS[key])
            ax.fill_between(trace['steps'],mean-std,mean+std,color=COLORS[key],alpha=.12,linewidth=0)
        ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=12)
        ax.set_xlabel('Outer training step',labelpad=10)
        ax.set_xticks([0,15,30,45,60,75]);ax.set_xlim(0,75)
        style_axes(ax);ax.grid(axis='y',color='#e6eaf0',linewidth=.8)
        if metric=='val_greedy_pass1':
            baseline=summary['base_model']['validation_benchmark']['greedy_pass1']*100
            ax.axhline(baseline,color=COLORS['base'],ls='--',lw=1)
            ax.set_ylim(0,100);ax.yaxis.set_major_formatter(PercentFormatter())
        else:ax.set_ylim(bottom=0)
    handles=[Line2D([0],[0],color=COLORS[k],lw=2,label=LABELS[k]) for k in GROUPS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.06),ncol=4,frameon=False,fontsize=10)
    fig.text(.075,.018,'Static GRPO has no generated pool. Pool growth measures stored unique pairs, not proof that the Mutator improved.',fontsize=9,color='#788598')
    save_figure(fig,output,'v2_training_evolution')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'reports/v2_experiment_summary.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'assets')
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text())
    data=prepare_plot_data(summary)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'nano-CodeSelfPlay-v2'}):
        plot_test(data,args.output_dir)
        plot_training(data,summary,args.output_dir)
    print(f'Rendered two figures in PNG and SVG from {args.summary}')


if __name__=='__main__':main()
