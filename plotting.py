#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import oemof.solph as solph
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import os
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

# register matplotlib converters which have been overwritten by pandas
register_matplotlib_converters()


#################################################################

def make_directory(folder_name):
    existing_folders = next(os.walk('.'))[1]
    if folder_name in existing_folders:
        print('----------------------------------------------------------')
        print('Folder "' + folder_name + '" already exists in current directory.')
        print('----------------------------------------------------------')
    else:
        path = "./" + folder_name
        os.mkdir(path)
        print('----------------------------------------------------------')
        print('Created folder "' + folder_name + '" in current directory.')
        print('----------------------------------------------------------')


def adjust_yaxis(ax, ydif, v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def extract_results(model, approach, **kwargs):
    """ Extract data from Pyomo variables in DataFrames and plot for visualization.

    Extract the results from the toy model.
    A distinction for the different approaches has to be made since
    the demand response variables and the way they are handled vary.

    :param model: oemof.solph.models.Model
        The solved optimization model (including results)
    :param approach: str
        Must be one of ["DIW", "IER", "DLR", "TUD"]
    :return: df_model: pd.DataFrame
        A pd.DataFrame containing the concatenated and renamed results sequences
    """

    invest = kwargs.get('invest', False)
    # Normalized indicating whether to use normalized data for dispatch modeling
    # in investment modeling, data is always normalized
    normalized = kwargs.get('normalized', False)

    # ########################### Get DataFrame out of Pyomo and rename series

    # Determine which generation results to extract
    include_coal = kwargs.get('include_coal', True)
    include_gas = kwargs.get('include_gas', False)

    # Introduce shortcuts
    bus_elec_seqs = solph.views.node(model.es.results['main'], 'bus_elec')['sequences']
    dsm_seqs = solph.views.node(model.es.results['main'], 'demand_dsm')['sequences']

    if invest:
        # investment results are stored as a pd.Series
        # For multiple DR units, they iteratively have to be accessed;
        # only one unit considered here for now
        dsm_invest = solph.views.node(
            model.es.results['main'],
            'demand_dsm')['scalars'].values[0]

    # Generators coal
    if include_coal:
        df_coal_1 = bus_elec_seqs[
            (('pp_coal_1', 'bus_elec'), 'flow')].rename('coal1', inplace=True)
    else:
        df_coal_1 = pd.Series(index=bus_elec_seqs.index)

    if include_gas:
        df_gas_1 = bus_elec_seqs[
            (('pp_gas_1', 'bus_elec'), 'flow')].rename('gas1', inplace=True)
    else:
        df_gas_1 = pd.Series(index=bus_elec_seqs.index)

    # Generators RE
    df_wind = bus_elec_seqs[
        (('wind', 'bus_elec'), 'flow')].rename('wind', inplace=True)

    # Shortage/Excess
    df_shortage = bus_elec_seqs[
        (('shortage_el', 'bus_elec'), 'flow')].rename('shortage', inplace=True)

    df_excess = bus_elec_seqs[
        (('bus_elec', 'excess_el'), 'flow')].rename('excess', inplace=True)

    # ---------------- Extract DSM results (all approaches) ---------------------
    # Parts of results extraction is dependent on kwargs (might be removed later)

    # Demand after DSM
    df_demand_dsm = bus_elec_seqs[
        (('bus_elec', 'demand_dsm'), 'flow')].rename('demand_dsm',
                                                     inplace=True)

    # Downwards shifts (shifting)
    df_dsmdo_shift = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                      == 'dsm_do_shift'].sum(
        axis=1).rename('dsm_do_shift', inplace=True)

    # Downwards shifts (shedding)
    df_dsmdo_shed = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                     == 'dsm_do_shed'].sum(
        axis=1).rename('dsm_do_shed', inplace=True)

    # Upwards shifts
    df_dsmup = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                == 'dsm_up'].sum(
        axis=1).rename('dsm_up', inplace=True)

    df_dsm_add = None

    # Get additional DSM results dependent on approach considered
    if approach == "DLR":
        # Original shift values
        df_dsmdo_orig = df_dsmdo_shift.copy().rename('dsm_do_orig',
                                                     inplace=True)
        df_dsmup_orig = df_dsmup.copy().rename('dsm_up_orig',
                                               inplace=True)

        # Balacing values
        df_dsmdo_bal = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                    == 'balance_dsm_do'].sum(
            axis=1).rename('balance_dsm_do', inplace=True)
        df_dsmup_bal = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                    == 'balance_dsm_up'].sum(
            axis=1).rename('balance_dsm_up', inplace=True)

        # DSM storage levels
        df_dsmsldo = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                        == 'dsm_do_level'].sum(
            axis=1).rename('dsm_sl_do', inplace=True)
        df_dsmslup = dsm_seqs.iloc[:, dsm_seqs.columns.str[1]
                                        == 'dsm_up_level'].sum(
            axis=1).rename('dsm_sl_up', inplace=True)

        df_dsmdo_shift = df_dsmdo_orig.add(df_dsmup_bal).rename('dsm_do_shift',
                                                                inplace=True)
        df_dsmup = df_dsmup_orig.add(df_dsmdo_bal).rename('dsm_up',
                                                          inplace=True)

        df_dsm_add = pd.concat([df_dsmdo_orig, df_dsmup_orig,
                                df_dsmdo_bal, df_dsmup_bal,
                                df_dsmsldo, df_dsmslup], axis=1)

    # Effective DSM shift (shifting only)
    df_dsm_tot = df_dsmdo_shift - df_dsmup
    df_dsm_tot.rename('dsm_tot', inplace=True)

    # DSM storage level
    df_dsm_acum = df_dsm_tot.cumsum()
    df_dsm_acum.rename('dsm_acum', inplace=True)

    # Original demand before DSM
    df_demand_el = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].demand
    df_demand_el.rename('demand_el', inplace=True)

    # Capacity limit for upshift
    df_capup = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].capacity_up
    df_capup.rename('cap_up', inplace=True)

    # Capacity limit for downshift
    df_capdo = [_ for _ in model.NODES.data() if str(_) == 'demand_dsm'][0].capacity_down
    df_capdo.rename('cap_do', inplace=True)

    if invest:
        df_demand_el = df_demand_el.mul(dsm_invest)
        df_capup = df_capup.mul(dsm_invest)
        df_capdo = df_capdo.mul(dsm_invest)

    elif normalized:
        df_demand_el = df_demand_el.mul(kwargs.get('max_demand', 1))
        df_capup = df_capup.mul(kwargs.get('max_capacity_up', 1))
        df_capdo = df_capdo.mul(kwargs.get('max_capacity_down', 1))

    # ####### Merge all data into one DataFrame
    df_model = pd.concat([df_coal_1, df_wind, df_excess, df_shortage,
                          df_demand_dsm, df_dsmdo_shift, df_dsmdo_shed, df_dsmup,
                          df_dsm_tot, df_dsm_acum, df_demand_el,
                          df_capup, df_capdo],
                         axis=1)

    # Add additional dsm values for certain approaches
    if df_dsm_add is not None:
        df_model = pd.concat([df_model, df_dsm_add], axis=1, sort=False)

    if invest:
        return df_model, dsm_invest
    else:
        return df_model


def plot_dsm(df_gesamt, directory, project, days, **kwargs):
    """ Create a plot of DSM activity """
    figsize = kwargs.get('figsize', (15, 10))
    save = kwargs.get('save', False)
    approach = kwargs.get('approach', None)
    include_approach = kwargs.get('include_approach', False)
    include_generators = kwargs.get('include_generators', False)
    ax1_ylim = kwargs.get('ax1_ylim', [-10, 250])
    ax2_ylim = kwargs.get('ax2_ylim', [-110, 150])

    # ############ DATA PREPARATION FOR FIGURE #############################

    # Create Figure
    for info, slice in df_gesamt.resample(str(days) + 'D'):

        # Generators from model
        # hierarchy for plot: wind, coal, gas, shortage
        if include_generators:
            graph_wind = slice.wind.values
            graph_coal = graph_wind + slice.coal1.values
            graph_gas = graph_coal + slice.gas1.values
            graph_shortage = graph_gas + slice.shortage.values

        #################
        # first axis
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_ylim(ax1_ylim)

        # x-Axis date format
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
        ax1.set_xlim(info - pd.Timedelta(1, 'h'), info + pd.Timedelta(days * 24 + 1, 'h'))
        plt.xticks(pd.date_range(start=info._date_repr, periods=days * 24, freq='H'), rotation=90)

        # Demands
        # ax1.plot(range(timesteps), dsm, label='demand_DSM', color='black')
        ax1.step(slice.index, slice.demand_el.values, where='post', label='Demand', linestyle='--', color='blue')
        ax1.step(slice.index, slice.demand_dsm.values, where='post', label='Demand after DSM', color='black')

        # DSM Capacity
        ax1.step(slice.index, slice.demand_el + slice.cap_up, where='post', label='DSM Capacity', color='red',
                 linestyle='--')
        ax1.step(slice.index, slice.demand_el - slice.cap_do, where='post', color='red', linestyle='--')

        # Generators
        if include_generators:
            ax1.fill_between(slice.index, 0, graph_wind, step='post', label='Wind', facecolor='darkcyan', alpha=0.5)
            ax1.fill_between(slice.index, graph_wind, graph_coal, step='post', label='Coal', facecolor='black', alpha=0.5)
            ax1.fill_between(slice.index, graph_coal, graph_gas, step='post', label='Gas', facecolor='brown', alpha=0.5)

        ax1.legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

        # plt.xticks(range(0,timesteps,5))

        plt.grid()

        ###########################
        # Second axis
        ax2 = ax1.twinx()
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
        ax2.set_xlim(info - pd.Timedelta(1, 'h'), info + pd.Timedelta(days * 24 + 1, 'h'))
        plt.xticks(pd.date_range(start=info._date_repr, periods=days * 24, freq='H'), rotation=90)

        ax2.set_ylim(ax2_ylim)

        ax2.fill_between(slice.index, 0, -slice.dsm_do_shift,
                         step='post',
                         label='DSM_down_shift',
                         facecolor='red',
                         # hatch='.',
                         alpha=0.3)
        ax2.fill_between(slice.index, -slice.dsm_do_shift,
                         -(slice.dsm_do_shift + slice.dsm_do_shed),
                         step='post',
                         label='DSM_down_shed',
                         facecolor='blue',
                         # hatch='.',
                         alpha=0.3)
        ax2.fill_between(slice.index, 0, slice.dsm_up,
                         step='post',
                         label='DSM_up',
                         facecolor='green',
                         # hatch='.',
                         alpha=0.3)
        ax2.plot(slice.index, slice.dsm_acum,
                 linestyle='none',
                 markersize=8,
                 marker="D",
                 color="dimgrey",
                 fillstyle='none',
                 drawstyle="steps-post",
                 label='DSM acum',
                 )

        # Legend axis 2
        ax2.legend(bbox_to_anchor=(0., -0.3, 1., 0.102), loc=3, ncol=3, borderaxespad=0., mode="expand")
        ax1.set_xlabel('Time t in h')
        ax1.set_ylabel('MW')
        ax2.set_ylabel('$\Delta$ MW')

        if approach is not None:
            plt.title(approach)

        plt.show()

        if save:
            fig.set_tight_layout(True)
            name = 'Plot_' + project + '_' + info._date_repr + '.png'
            if include_approach:
                name = 'Plot_' + project + '_' + approach + '_' + info._date_repr + '.png'
            fig.savefig(directory + 'graphics/' + name)
            plt.close()
            print(name + ' saved.')


def plot_case(data, case='constant', **kwargs):
    """ Function to plot the case considered.

    Case is defined by availability time series, i.e. capacity bounds for DSM and
    demand before DSM as well as generation pattern.
    """
    show = kwargs.get('show', True)
    save_figs = kwargs.get('save_figs', False)

    # Plot demand, wind generation and DR capacity limits
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(111)

    _ = plt.title('Generation and demand for case "' + case + '"')

    # Define xaxis ticks
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
    ax.set_xlim(data.index.values[0] - pd.Timedelta(1, 'h'),
                data.index.values[0] + pd.Timedelta(1, 'h'))
    plt.xticks(pd.date_range(start=data.index.values[0],
                             periods=len(data) + 1,
                             freq='H'), rotation=90)

    ax.plot(data.index, data['demand_el'].values, drawstyle="steps-post",
            label="demand")
    ax.plot(data.index, data['wind'].values, drawstyle="steps-post",
            label="generation")

    # Cap_up and Cap_do only included for proper alignment here
    ax.plot(data.index, (data['demand_el'] + data['Cap_up']).values,
            drawstyle="steps-post", color="limegreen", label="upper limit")
    ax.plot(data.index, (data['demand_el'] - data['Cap_do']).values,
            drawstyle="steps-post", color="lightcoral", label="lower limit")

    _ = ax.set_yticks(
        range(-(data.Cap_do.max() - 100), data.Cap_up.max() + 125, 25))
    ax.legend(bbox_to_anchor=(0., -0.5, 1., 0.102), loc=2, ncol=2,
              borderaxespad=0.)
    _ = ax.set_xlabel("Time in h")
    _ = ax.set_ylabel("capacity in MW \n(demand, generation,\n abs. limits)")

    plt.grid(alpha=0.6)

    # Delta MW on secondary y_axis
    ax2 = ax.twinx()
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
    ax2.set_xlim(data.index.values[0] - pd.Timedelta(1, 'h'),
                 data.index.values[-1] + pd.Timedelta(1, 'h'))
    plt.xticks(pd.date_range(start=data.index.values[0],
                             periods=len(data) + 1,
                             freq='H'), rotation=90)

    ax2.plot(data.index, data.Cap_up.values, drawstyle="steps-post",
             # secondary_y=True,
             linestyle=":", color="darkgreen", label="Cap_up (right axis)")
    ax2.plot(data.index, (data.Cap_do * -1).values, drawstyle="steps-post",
             # secondary_y=True,
             linestyle=":", color="saddlebrown", label="Cap_do (right axis)")

    _ = ax2.set_yticks(range(-data.Cap_do.max(), data.Cap_up.max() + 50, 50))
    ax2.legend(bbox_to_anchor=(0., -0.5, 1., 0.102), loc=1, ncol=1,
               borderaxespad=0.)
    _ = ax2.set_ylabel("difference $\Delta$ MW \n(Cap_up, Cap_do)")  #

    # Do axis aligment
    align_yaxis(ax, -(data.Cap_do.max() - 100), ax2, -data.Cap_do.max())
    align_yaxis(ax, data.Cap_up.max() + 100, ax2, data.Cap_up.max())

    if show:
        plt.show()

    if save_figs:
        name = 'toy-model_' + case + '.png'
        fig.savefig('./graphics/' + name)
        plt.close()
        print(name + " saved.")


def plot_case_residual(data, case='constant', **kwargs):
    """Plot the residual load for the respective case.

    Residual load is defined here as the difference between
    generic generation and demand, i.e., what is actually to be balanced.
    """

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(111)
    _ = plt.title('"Residual load" for case "' + case + '"')

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%d.%m - %H h'))  # ('%d.%m-%H h'))
    ax.set_xlim(data.index.values[0] - pd.Timedelta(1, 'h'),
                data.index.values[0] + pd.Timedelta(1, 'h'))
    plt.xticks(pd.date_range(start=data.index.values[0],
                             periods=len(data) + 1,
                             freq='H'), rotation=90)

    ax.plot(data.index, (data['wind'] - data['demand_el']).values,
            drawstyle="steps-post",
            linestyle="-.", label="residual load", color="black")
    _ = ax.set_yticks(range(-100, 125, 25))
    plt.grid()
    _ = ax.set_ylabel("MW \n(residual load)")

    plt.show()