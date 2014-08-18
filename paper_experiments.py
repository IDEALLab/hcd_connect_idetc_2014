'''
    Code for method comparison experiment of HCD cases
    Used to compare method usage across different factors:
     - User - IDEO vs. Non-IDEO
     - Focus Area - Agriculture vs. Healthcare, etc.
    Mark Fuge 2014 with some code written by Bud Peters
    
    This experiment code is what is used to the produce the results in
    Mark Fuge, Alice Agogino,  "User Research Methods for Development Engineering: 
    A Study of Method Usage with IDEO's HCD Connect," for Proceedings of ASME 2014 
    International Design Engineering Technical Conferences & Computers and 
    Information in Engineering Conference, August 17-20, 2014, Buffalo, USA.
'''
import os
import csv
import numpy as np
import simplejson as json
from scipy.stats import scoreatpercentile as percentile
import scipy.stats as stats
from sklearn.utils import resample
from Queue import PriorityQueue as PQ
import matplotlib.pylab as plt
import prettyplotlib as ppl
from prettyplotlib.colors import set2,light_grey
import brewer2mpl

blues4=brewer2mpl.get_map('Blues', 'sequential', 4).mpl_colors
blues4.reverse()

set3=brewer2mpl.get_map('Set3', 'qualitative', 3).mpl_colors
set1=brewer2mpl.get_map('Set1', 'qualitative', 3).mpl_colors
dark2=brewer2mpl.get_map('Dark2', 'qualitative', 3).mpl_colors

bootstrap_num=10000
array_sample = lambda x: 100*np.array([np.average(resample(x),axis=0) for i in range(bootstrap_num)])

## Following code enables nicer plots

twocol=40.0/6 # ASME figure format two column is 40picas. /6 to get inches
onecol=20.0/6 # ASME figure format one column is 20picas. /6 to get inches
figdpi=120  # 600 dpi is ASME standard

from matplotlib.ticker import MaxNLocator,AutoLocator
almost_black = '#262626'
default_color = blues4[0]

def setfont(size=40):
    ''' Sets some fonts for plotting the figures
    '''
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size' : size}
    plt.matplotlib.rc('font', **font)

def fix_legend(ax=None,**kwargs):
    '''Applies nice coloring to legend'''
    if not ax:
        ax = plt.gca()
    light_grey = np.array([float(248)/float(255)]*3)
    legend = ax.legend(frameon=True,fontsize=16,**kwargs)
    ltext = ax.get_legend().get_texts()
    for lt in ltext:
        plt.setp(lt, color = almost_black)
    rect = legend.get_frame()
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)
    # Change the legend label colors to almost black, too
    texts = legend.texts
    for t in texts:
        t.set_color(almost_black)
    
def fix_axes(ax=None):
    '''
    Removes top and left boxes
    Lightens text
    '''
    if not ax:
        ax = plt.gca()
    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower'))
    # Change the labels to the off-black
    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)

## Back to Experiment code now

def load_hcd_cases(data_path):
    ''' Loads case study data for stories, methods, and cases
        Assumes that methods and stories are correctly ordered by line number
    '''
    # Load Stories into an indexed array table
    story_csv = csv.reader(open(data_path+'stories.csv'),delimiter = '|')
    method_csv = csv.reader(open(data_path+'methods.csv'),delimiter = '|')
    case_csv = csv.reader(open(data_path+'cases.csv'),delimiter = '|')
    
    stories=[]
    case_categories=[]
    for story in story_csv:
        stories.append((story[1].lower(),story[2].lower()))
        # Focus Area + User
        uid = ['IDEO' if story[5]=='IDEO.org' else 'noIDEO' ]
        cats = [story[8].lower().split(';'),uid]
        case_categories.append(cats)
    methods={}
    for method in method_csv:
        methods[int(method[0])]=[method[1],method[2]]
    methods = methods.values()
    cases=np.zeros((len(stories),len(methods))) 
    for story_id,method_id in case_csv:
        cases[int(story_id)][int(method_id)]=1
    # Now remove invalid cases:
    ft=np.array([True if len(s[1])>6 else False for s in stories])
    return (np.array(stories)[ft],
            np.array(methods),
            np.array(cases)[ft],
            np.array(case_categories)[ft])

def get_method_bounds(X):
    sample_counts = array_sample(X)
    percentsat = lambda x,k: np.array([percentile(x[:,i],k) for i,m in enumerate(xlabels)])
    median=percentsat(sample_counts,50)
    yerr = np.vstack([median-percentsat(sample_counts,05),
                      percentsat(sample_counts,95)-median]).transpose()
    return median,yerr   
    
def method_bar(median,yerr,xlabels, **kwargs):
    colors=kwargs.pop('colors',default_color)
    legend=kwargs.pop('legend',None)
    x_offset=kwargs.pop('x_offset',0.0)
    label_x_offset = kwargs.pop('label_x_offset',0.5)
    rotation = kwargs.pop('rotation',90)
    annotatefontsize = kwargs.pop('annotatefontsize',5)
    xfontsize = kwargs.pop('xfontsize',6)
    width=kwargs.pop('width',0.7)
    bar=ppl.bar(ax,np.array(range(len(xlabels)))+x_offset,median,width=width,
            yerr=[yerr[:,0],yerr[:,1]], ecolor='0.5',
            grid='y', capsize=0,color=colors,
            annotate=True,annotatefmt='%.d',annotatefontsize=annotatefontsize,
            **kwargs )
    plt.ylabel('% cases used')
    ax.set_xticks(np.array(range(len(xlabels)))+label_x_offset)
    ax.set_xticklabels(xlabels,fontsize=xfontsize,rotation=rotation,va='top',ha='right')
    if legend:
        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)
    
def method_boxplot(data,xlabels, line_color=default_color,
                   med_color=None, legend=None, x_offset=0.0):
    if not med_color:
        med_color=line_color
    ax.grid(axis='x', color='0.9', linestyle='-', linewidth=0.2)
    ax.set_axisbelow(True)
    ppl.boxplot(ax,data, positions=np.array(range(len(xlabels)))-x_offset,
                yticklabels=xlabels.tolist(),linewidth=0.3,
                widths=0.2,show_caps=False,sym='',vert=False,
                line_color=line_color, med_color=med_color)
    plt.xlabel('% cases used')
    ax.set_yticks(np.array(range(len(xlabels))))
    ax.set_yticklabels(xlabels,fontsize=6)
    spines_to_remove = ['top', 'right','left']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    
    if legend:
        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)
        
def method_errorbar(data,xlabels, line_color=default_color,
                   med_color=None, legend=None, y_offset=0.0,alpha=0.05):
    if not med_color:
        med_color=line_color
    ax.grid(axis='x', color='0.9', linestyle='-', linewidth=0.2)
    ax.set_axisbelow(True)
    n,m=data.shape
    medians = [percentile(data[:,i],50) for i in range(m)]
    xerr = [[ medians[i]-percentile(data[:,i],100*(alpha/2.)),
              percentile(data[:,i],100*(1-alpha/2.))-medians[i] ]
             for i in range(m)]
    xerr=np.array(xerr).transpose()
    y_marks = np.array(range(len(xlabels)))-y_offset
    plt.errorbar(y=y_marks,
                 x=medians,xerr=xerr,fmt='|',capsize=0,color=line_color,
                 ecolor=line_color,elinewidth=0.3,markersize=2)
    plt.xlabel('% cases used', fontsize=8)
    ax.tick_params(axis='x', which='both', labelsize=8)
    ax.set_yticks(np.array(range(len(xlabels))))
    ax.set_yticklabels(xlabels,fontsize=6)
    plt.ylim((min(y_marks)-0.5,max(y_marks)+0.5))
    spines_to_remove = ['top', 'right','left']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ppl.utils.remove_chartjunk(ax, ['top', 'right', 'bottom'], show_ticks=False)
    if legend:
        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)
        
def plot_boolean_frequency(data,labels,**kwargs):
    alpha=0.05
    boolean_percent = lambda X: np.count_nonzero(X)/float(len(X))
    boolean_sample = lambda x: np.array([boolean_percent(resample(x)) for i in range(bootstrap_num)])
    medians=[]
    yerr=[]
    for d,l in zip(data,labels):
        d_samples = boolean_sample(d)
        low=percentile(d_samples,100*alpha/2.)
        med=percentile(d_samples,50)
        high=percentile(d_samples,100*(1-alpha/2.))
        print '[%.2f,%.2f,%.2f]:%s'%(low,med,high,l)
        medians.append(med)
        yerr.append([med-low,high-med])
    yerr=np.array(yerr)
    kwargs['width']=0.2
    kwargs['xfontsize']=10
    method_bar(medians,yerr,labels,**kwargs)
    plt.ylim(0,1)
 
def plot_method_pairs_and_matrix(case_studies,fileappend=''):
    case_cov=np.cov(case_studies.transpose())
    case_corr=np.corrcoef(case_studies.transpose())
    cmatrix= case_corr
    fig = plt.figure(figsize=(twocol,twocol),dpi=figdpi,tight_layout=True)
    ax = fig.add_subplot(111)
    ppl.pcolormesh(fig,ax,cmatrix[inds][:,inds],#-np.diag([.99]*len(meths)),
                   yticklabels=np.array(mindex)[inds].tolist(),
                   xticklabels=np.array(mindex)[inds].tolist(),
                   cmap=ppl.mpl.cm.RdBu,vmax=0.4,vmin= -0.4)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    cm=dark2
    [l.set_color(cm[m_codes[mindex.index(l.get_text())]]) for i,l in enumerate(ax.get_yticklabels())]
    [l.set_color(cm[m_codes[mindex.index(l.get_text())]]) for i,l in enumerate(ax.get_xticklabels())]
    fig.show()
    fig.savefig(figure_path+('method_matrix%s.pdf'%fileappend))
    
    #Show the highly correlated methods
    pq=PQ()
    pq_cross=PQ()
    for i in range(len(meths)):
        for j in range(i+1,len(meths)):
            m1text='(%s) %s'%(meths[i,1],meths[i,0])
            m2text='(%s) %s'%(meths[j,1],meths[j,0])
            pq.put((-cmatrix[i,j],(m1text,m2text)))
            if meths[i,1]!= meths[j,1]:
                pq_cross.put((-cmatrix[i,j],(m1text,m2text)))
    
    # Output the method correlations
    # Sets how many highly correlated methods should be displayed
    print_cap = 20
    moutfile = open(results_path+('method_corrs%s.csv'%fileappend),'w')
    print 'All methods:'
    for i in range(pq.qsize()):
        v,(m1,m2)=pq.get()
        if i < print_cap:
            print '%.2f & %s & %s\\\\'%(-v,m1,m2)
        moutfile.write('%.9f | %s | %s\n'%(-v,m1,m2))
    moutfile.close()
    
    moutfile = open(results_path+('method_corrs_cross%s.csv'%fileappend),'w')
    print 'Just cross methods:'
    for i in range(pq_cross.qsize()):
        v,(m1,m2)=pq_cross.get()
        if i < print_cap:
            print '%.2f & %s & %s\\\\'%(-v,m1,m2)
        moutfile.write('%.9f | %s | %s\n'%(-v,m1,m2))
    moutfile.close()
        
if __name__ == "__main__":
    print_output = True
    plot_output = True
    setfont(15)
    
    # Take care of filepaths
    data_path='./'
    results_path='results/'
    figure_path='figures/'
    # Check if data directory exists, if not, create it
    for path in [data_path, results_path, figure_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    # Now load the actual data
    story_text, methods, cases,case_categories = load_hcd_cases(data_path)
    
    # There is a data error in the first case -- it splits community development
    # into two categories. Since this is the only case that does this, I'll go 
    # ahead and merge the two together.
    case_categories[0][0].remove('development')
    case_categories[0][0].remove('community')
    case_categories[0][0].append('community development')
    
    
    # Now take the cases and divide them up into the different slices:
    # First, Focus Area:
    focus_area_names = list(set(np.concatenate(case_categories[:,0].flatten())))
    focus_area_names = np.array(focus_area_names)
    focus_areas = np.zeros(shape=(len(case_categories),len(focus_area_names)),dtype='int8')
    for i,cats in enumerate(case_categories[:,0]):
        for cat in cats:
            j=int(np.argwhere(cat==focus_area_names))
            focus_areas[i,j]=1
    # List out most popular focus areas:
    focus_counts = np.sum(focus_areas,axis=0)
    focus_combined = np.vstack([focus_counts,focus_area_names]).transpose()
    sorted_focus_areas = np.argsort(np.sum(focus_areas,axis=0))[::-1]
    print "Most common focus areas by number of cases:"
    print focus_combined[sorted_focus_areas]
    for num, focus in focus_combined[sorted_focus_areas]:
        print "%d & %.1f & %s\\\\"%(int(num),100*(int(num)/float(len(cases))),focus.title())
    
    # Now make individual method comparison plots
    # Overall Plot of method usage:
    X = cases
    xlabels=methods[:,0]
    mlabels=methods[:,1]
    #cm=blues4
    cm=dark2
    colors = 0*(mlabels=='H')+1*(mlabels=='C')+2*(mlabels=='D')
    colors = np.array([cm[c] for c in colors])
    median,yerr = get_method_bounds(X)
    sorted_methods = np.argsort(median)[::-1]
    
    fig = plt.figure('Method Usage',figsize=(twocol,onecol),dpi=figdpi,tight_layout=True)
    ax = fig.add_subplot(111)
    p1 = plt.Rectangle((0, 0), 1, 1, fc=cm[0],lw=0)
    p2 = plt.Rectangle((0, 0), 1, 1, fc=cm[1],lw=0)
    p3 = plt.Rectangle((0, 0), 1, 1, fc=cm[2],lw=0)
    legend=ax.legend((p1, p2, p3), ('Hear','Create','Deliver'),fontsize=15)
    method_bar(median[sorted_methods],yerr[sorted_methods,:],
               xlabels[sorted_methods], colors=colors[sorted_methods],legend=legend)
    fig.autofmt_xdate()
    plt.title('Method Usage: Overall')
    fig.show()
    fig.savefig(figure_path+'freq_overall.pdf')
    
    # Change ordering for correct vertical order
    cm=blues4
    colors = 0*(mlabels=='H')+1*(mlabels=='C')+2*(mlabels=='D')
    colors = np.array([dark2[c] for c in colors])
    setfont(10)
    sorted_methods=sorted_methods[::-1]
    
    # Now make plots for each focus area:
    for i,focus in enumerate(focus_area_names):
        # Get just the cases that used this focus area
        f_ind=focus_areas[:,i]==1
        X=cases[f_ind,:]
        otherX=cases[f_ind==False,:]
        fig = plt.figure('Method Usage: %s'%focus,figsize=(onecol,1.15*onecol),dpi=figdpi,tight_layout=True)
        ax = fig.add_subplot(111)
        hB, = plt.plot([-5,-5],'-',color=cm[0])
        hR, = plt.plot([-5,-5],'-',color=cm[1])
        legend = ax.legend((hB, hR),('Other', focus[0:15]),loc=4,fontsize=8,
                           borderaxespad=0.)
        # Plot the overall method frequency
        sample_counts = array_sample(otherX)
        method_errorbar(sample_counts[:,sorted_methods],
                       xlabels=xlabels[sorted_methods],
                       line_color=cm[0],legend=legend)
        # Plot the focus area method frequency
        fsample_counts = array_sample(X)
        method_errorbar(fsample_counts[:,sorted_methods],
                       xlabels=xlabels[sorted_methods],
                       line_color=cm[1],y_offset=0.25)
        plt.title('Method Usage:\n%s'%focus,fontsize=10)
        [l.set_color(colors[sorted_methods][i]) for i,l in enumerate(ax.get_yticklabels())]
        #fig.show()
        fig.savefig(figure_path+'freq_%s.pdf'%focus)
        
    p_stats=[]     
    t_stats=[]
    for i,focus in enumerate(focus_area_names):
        # Get just the cases that used this focus area
        f_ind=focus_areas[:,i]==1
        X=cases[f_ind,:]
        otherX=cases[f_ind==False,:]
        mean_X = np.average(X,axis=0)
        mean_otherX = np.average(otherX,axis=0)
        differences = mean_X-mean_otherX
        stds = np.std(X,axis=0)
        t,p=stats.ttest_ind(X,otherX,equal_var=False)
        for i,prob in enumerate(p):
            p_stats.append((prob,differences[i],focus,methods[i,0]))
        for i,tstat in enumerate(t):
            t_stats.append((tstat,differences[i],focus,methods[i,0]))
    
    dt = np.dtype([('num', np.float64),('difference', np.float64),
                   ('focus', str, 30), ('method', str, 50)])
    p_stats = np.array(p_stats,dtype=dt)
    t_stats = np.array(t_stats,dtype=dt)
    t_stats = np.sort(t_stats,order=['num'])
    
    # Now do the probability plot
    fig,ax=plt.subplots(1,tight_layout=True)
    setfont(20)
    ((osm,osr),(slope,intercept,r)) = stats.probplot(t_stats['num'], dist="norm",fit=True, plot=plt)
    stat_dump=np.vstack([osr,osm,t_stats['focus'],t_stats['method']]).T
    stat_dump=[{'t':float(s[0]),'q':float(s[1]),
                'focus':s[2],'method':s[3]} for s in stat_dump.tolist()]
    
    json.dump(stat_dump,open(results_path+'method_t_stats.json','wb'))
    plt.setp(ax.lines[0],'markeredgewidth',0.0)
    ppl.utils.remove_chartjunk(ax, ['top', 'right'], show_ticks=False)
    fig.show()
    fig.savefig(figure_path+'focus_area_QQ.pdf')
    
    # Do Benjamini Hochberg Procedure:
    alph=0.05 # Sets false discovery rate
    m=len(p_stats)
    c=1
    # Uncomment the below line to run the Bejamini-Hochberg-Yekutieli Procedure instead
    # That procedure assumes arbitrary dependence between the tests, rather than indep.
    #c = np.sum([1./k for k in range (1,m+1)])
    BH_cutoffs=[k*alph/float(c*m) for k in range(1,m+1)]
    sorted_p = np.sort(p_stats)
    discoveries = sorted_p['num'] < BH_cutoffs
    for prob, diff, focus, method in  sorted_p[discoveries]:
        print "%.1e & %.1f & %s & %s\\\\"%(prob,diff*100,method.title(),focus.title())
    
    # Now compare IDEO stuff
    setfont(15)
    ideo_names = np.array(['IDEO','NoIDEO'])
    names=np.array([n[0]  for n in case_categories[:,1]])
    ideo_ind = names=='IDEO'
    nideo_ind = names!='IDEO'
    m_codes = 0*(mlabels=='H')+1*(mlabels=='C')+2*(mlabels=='D')
    ideosample_counts = array_sample(cases[ideo_ind,:])
    nideosample_counts = array_sample(cases[nideo_ind,:])
    ideo_averages = np.average(ideosample_counts,axis=0)
    dt=np.dtype([('category', np.float64),('average', np.float64)])
    t=np.vstack([-m_codes,ideo_averages]).transpose()
    comb_rank = np.array([(k[0],k[1]) for k in t],dtype=dt)
    ideo_sort=np.argsort(comb_rank,order=('category','average'))
    fig = plt.figure('Method Usage: IDEO vs. Non-IDEO',figsize=(onecol,1.15*onecol),dpi=figdpi,tight_layout=True)
    ax = fig.add_subplot(111)
    ideo_col=cm[0]
    nideo_col=cm[1]
    hideo, = plt.plot([-5,-5],'-',color=ideo_col)
    hnideo, = plt.plot([-5,-5],'-',color=nideo_col)
    legend = ax.legend((hideo, hnideo),('IDEO', 'Non-IDEO'),loc=4,fontsize=8)
    hB.set_visible(False)
    hR.set_visible(False)
    method_errorbar(ideosample_counts[:,ideo_sort],
                   xlabels=xlabels[ideo_sort],
                   line_color=ideo_col,legend=legend)
    method_errorbar(nideosample_counts[:,ideo_sort],
                   xlabels=xlabels[ideo_sort],
                   line_color=nideo_col,y_offset=0.25, legend=legend)

    plt.title('Method Usage:\nIDEO vs. Non-IDEO',fontsize=8)
    [l.set_color(colors[ideo_sort][i]) for i,l in enumerate(ax.get_yticklabels())]
    fig.show()
    fig.savefig(figure_path+'freq_IDEO_sort.pdf')
    
    
    hear_ids=methods[:,1]=='H'
    connect_ids=methods[:,1]=='C'
    deliver_ids=methods[:,1]=='D'
    
    has_hear=np.sum(cases[:,hear_ids],axis=1)>0
    has_connect=np.sum(cases[:,connect_ids],axis=1)>0
    has_deliver=np.sum(cases[:,deliver_ids],axis=1)>0
    labels=['Hear','Create','Deliver']
    data=[has_hear,has_connect,has_deliver]
    print "Overall category usage"
    plot_boolean_frequency(data,labels)
    data=[has_hear[ideo_ind],has_connect[ideo_ind],has_deliver[ideo_ind]]
    print "IDEO category usage"
    plot_boolean_frequency(data,labels)
    data=[has_hear[nideo_ind],has_connect[nideo_ind],has_deliver[nideo_ind]]
    print "Non-IDEO category usage"
    plot_boolean_frequency(data,labels)
    
    
    labels=['Hear+Create','Hear+Deliver','Create+Deliver']
    data=[has_hear&has_connect,has_hear&has_deliver,has_connect&has_deliver]
    print "Overall category usage"
    plot_boolean_frequency(data,labels)
    data=[(has_hear&has_connect)[ideo_ind],
          (has_hear&has_deliver)[ideo_ind],
          (has_connect&has_deliver)[ideo_ind]]
    print "IDEO category usage"
    plot_boolean_frequency(data,labels)
    data=[(has_hear&has_connect)[nideo_ind],
          (has_hear&has_deliver)[nideo_ind],
          (has_connect&has_deliver)[nideo_ind]]
    print "Non-IDEO category usage"
    plot_boolean_frequency(data,labels)
    
    labels=['Hear+Create+Deliver']
    data=[has_hear&has_connect&has_deliver]
    print "Overall category usage"
    plot_boolean_frequency(data,labels)
    data=[(has_hear&has_connect&has_deliver)[ideo_ind],
          ]
    print "IDEO category usage"
    plot_boolean_frequency(data,labels)
    data=[(has_hear&has_connect&has_deliver)[nideo_ind],
          ]
    print "Non-IDEO category usage"
    plot_boolean_frequency(data,labels)
    
    cm=blues4
    fig = plt.figure('Category Usage',figsize=(twocol,onecol),dpi=figdpi,tight_layout=True)
    ax = fig.add_subplot(111)
    p1 = plt.Rectangle((0, 0), 1, 1, fc=cm[0],lw=0)
    p2 = plt.Rectangle((0, 0), 1, 1, fc=cm[1],lw=0)
    p3 = plt.Rectangle((0, 0), 1, 1, fc=cm[2],lw=0)
    p4 = plt.Rectangle((0, 0), 1, 1, fc=cm[0],lw=0)
    p5 = plt.Rectangle((0, 0), 1, 1, fc=cm[1],lw=0)
    legend=ax.legend((p4,p5),
                     ('IDEO','Non-IDEO'),fontsize=15)
    labels=['Hear','Create','Deliver','Hear+Create','Hear+Deliver',
            'Create+Deliver','Hear+Create+Deliver']
    data=[has_hear[ideo_ind],has_connect[ideo_ind],has_deliver[ideo_ind],
          (has_hear&has_connect)[ideo_ind],(has_hear&has_deliver)[ideo_ind],
          (has_connect&has_deliver)[ideo_ind],
          (has_hear&has_connect&has_deliver)[ideo_ind]
         ]
    plot_boolean_frequency(data,labels,colors=cm[0],legend=legend)
    data=[has_hear[nideo_ind],has_connect[nideo_ind],has_deliver[nideo_ind],
          (has_hear&has_connect)[nideo_ind],(has_hear&has_deliver)[nideo_ind],
          (has_connect&has_deliver)[nideo_ind],
          (has_hear&has_connect&has_deliver)[nideo_ind]
         ]
    plot_boolean_frequency(data,labels,colors=cm[1],legend=legend,x_offset=0.3,
                           rotation=20,label_x_offset=0.3-0.05)
    plt.title('Category Usage: Overall')
    xmin,xmax=plt.xlim()
    plt.xlim((-0.2,xmax-0.2))
    fig.show()
    fig.savefig(figure_path+'categories_overall.pdf')
    
    # Get Method pairs
    imethod_cols=np.sum(cases[ideo_ind],axis=0)>0
    meths = methods[imethod_cols]
    mindex=meths[:,0].tolist()
    m_codes=[0]*(meths[:,1]=='H')+[1]*(meths[:,1]=='C')+[2]*(meths[:,1]=='D')
    blues = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap
    inds=np.argsort(m_codes)
    sub_cases=cases[:,imethod_cols]
    plot_method_pairs_and_matrix(sub_cases)
    
    # As an additional check, I'll segment the cases that just use more than one phase
    # This is an additional set of experimental results that did not make it into the
    # original paper. Feel free to play around with it though, as it segments the
    # methods into slightly different (but still very interesting) categories.
    hc = has_hear&has_connect
    cd = has_connect&has_deliver
    hd = has_hear&has_deliver
    
    all_phase = has_hear&has_connect&has_deliver
    all_phase_cases = sub_cases[all_phase,:]
    plot_method_pairs_and_matrix(all_phase_cases,'_all_phase')
    
    multi_phase = hc|cd|hd
    multi_phase_cases = sub_cases[multi_phase,:]
    plot_method_pairs_and_matrix(multi_phase_cases,'_multi_phase')
    
    single_phase = ~multi_phase
    single_phase_cases = sub_cases[single_phase,:]
    plot_method_pairs_and_matrix(single_phase_cases,'_single_phase')
    