import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import my
import my.plot


def plot_follicle_and_ellipses(ax, transformed_mean_follicle, label_ellipses=False):
    ## Follicle
    ax.plot(
        [transformed_mean_follicle['x'].values.mean()],
        [transformed_mean_follicle['y'].values.mean()],
        marker='x', color='k', ls='none')


    ## Ellipses
    # C3
    ax.add_patch(matplotlib.patches.Ellipse(
        [-80, -70], 120, 200, fc='none', ec='gray'))
    
    # C2
    ax.add_patch(matplotlib.patches.Ellipse(
        [35, 35], 120, 200, fc='none', ec='gray'))
    
    # C1
    ax.add_patch(matplotlib.patches.Ellipse(
        [150, 140], 120, 200, fc='none', ec='gray'))
    
    
    ## Labels
    if label_ellipses:
        ax.text(-85, 60, 'C3', color='r', ha='center', va='center', size=14)
        ax.text(30, 160, 'C2', color='g', ha='center', va='center', size=14)
        ax.text(145, 270, 'C1', color='b', ha='center', va='center', size=14)
    

def consistent_limits(ax):
    ax.axis('image')
    ax.set_xlim((-300, 300))
    ax.set_ylim((300, -200))
    ax.set_xticks([])
    ax.set_yticks([])    
    
def normalize_and_blend_plot(
    masked_edge_data, evidence_data, edge_alpha=.3, ax=None,
    evidence_vmin=-1, evidence_vmax=1,
    x_index=None, y_index=None,
    ):
    
    ## Normalize and colormap the data
    # Normalize edge data to (0, 1) and colormap in black and white
    # This replaces masked data with the colormap's "bad value"
    edge_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    edge_data_rgba = plt.cm.gray_r(edge_norm(masked_edge_data))
    
    # Normalize evidence to (evidence_vmin, evidence_vmax) and 
    # colormap with custom_RdBu
    evidence_norm = matplotlib.colors.Normalize(
        vmin=evidence_vmin, vmax=evidence_vmax)
    evidence_data_rgba = my.plot.custom_RdBu_r()(
        evidence_norm(evidence_data))


    ## Alpha blend edge data and evidence data
    blended_rgba = my.plot.alpha_blend_with_mask(
        edge_data_rgba, 
        evidence_data_rgba, 
        edge_alpha,
        masked_edge_data.mask,
        )

    
    ## Plot
    im = my.plot.imshow(
        blended_rgba, ax=ax, 
        x=x_index, y=y_index)
    
    return im


def combine_whisker_occupancy_to_rgb(
    spatialized, full_spatial_bincenter_midx, 
    bins_x, bins_y,
    x_index, y_index,    
    vmin, vmax):
    
    ## Prepare occupancy for plotting
    # normalizing object
    norm = matplotlib.colors.Normalize(vmin, vmax)
    
    # Unstack whisker
    bgr = spatialized.unstack('whisker')
    
    # Reindex
    assert bgr.index.names == full_spatial_bincenter_midx.names
    bgr = bgr.reindex(full_spatial_bincenter_midx).fillna(0)
    
    # Separately colormap the data from each whisker using a unique color
    rgba_l = []
    for whisker in bgr.columns:
        # Build the colormap for this whisker
        color = {'C0': 'pink', 'C1': 'b', 'C2': 'g', 'C3': 'r'}[whisker]
        cmap = my.misc.CustomCmap(
            list(matplotlib.colors.to_rgb('white')),
            list(matplotlib.colors.to_rgb(color)),
            )
        
        
        ## Upsample occupancy_data to match edge_data
        # Extract data for this whisker and reshape into image
        occupancy_data = bgr[whisker].unstack('bin_x')
        
        # Convert bins to pixels (start pixel of bin)
        occupancy_data.index = pandas.Index(
            bins_y[occupancy_data.index.values], name='row')
        occupancy_data.columns = pandas.Index(
            bins_x[occupancy_data.columns.values], name='col')

        # Reindex like all_ces
        occupancy_data = occupancy_data.reindex(
            y_index).reindex(
            x_index, axis=1)

        # ffill (hence start pixel of bin above)
        occupancy_data = occupancy_data.ffill().ffill(axis=1)

        # shouldn't be any gaps, as long as first entry in evidence_data 
        # was upper left bin, which it should be because we reindexed by
        # full_spatial_bincenter_midx
        assert not occupancy_data.isnull().any().any()
        #~ assert evidence_data.shape == all_ces.shape


        ## Normalize and convert to rgba
        rgba = cmap(norm(occupancy_data.values))
        
        # Store
        rgba_l.append(rgba)

    # Mean the color data across whiskers
    meaned_rgba = np.array(rgba_l).mean(axis=0)
    
    # Strip out alpha
    meaned_rgb = meaned_rgba[:, :, :3]
    
    return meaned_rgb

def spatialize_evidence(
    evidence_data, keep_mask, full_spatial_bincenter_midx,
    bins_x, bins_y,
    x_index, y_index,
    ):
    """Convert evidence to a 2d spatial array"""

    # Combine evidence across whiskers, first within mouse, 
    # then across mice
    spatialized = evidence_data.mean(
        level=['mouse', 'bin_x', 'bin_y']).mean(
        level=['bin_x', 'bin_y'])

    # Mask
    spatialized = spatialized.reindex(keep_mask).dropna()
    assert list(spatialized.index.names) == ['bin_x', 'bin_y']

    # Reindex by full_spatial_bincenter_midx
    assert spatialized.index.names == full_spatial_bincenter_midx.names
    evidence_data = spatialized.reindex(
        full_spatial_bincenter_midx).fillna(0)

    # Unstack bin_x to spatialize
    evidence_data = evidence_data.unstack('bin_x')


    ## Upsample evidence_data to match edge_data
    # Convert bins to pixels (start pixel of bin)
    evidence_data.index = pandas.Index(
        bins_y[evidence_data.index.values], name='row')
    evidence_data.columns = pandas.Index(
        bins_x[evidence_data.columns.values], name='col')

    # Reindex like all_ces
    evidence_data = evidence_data.reindex(
        y_index).reindex(
        x_index, axis=1)

    # ffill (hence start pixel of bin above)
    evidence_data = evidence_data.ffill().ffill(axis=1)

    # shouldn't be any gaps, as long as first entry in evidence_data 
    # was upper left bin, which it should be because we reindexed by
    # full_spatial_bincenter_midx
    assert not evidence_data.isnull().any().any()
    #~ assert evidence_data.shape == all_ces.shape

    
    ## Return
    return evidence_data


def threshold_bins_by_n_whisks(sub_ae, mouse_thresh=4, nwpt_thresh=.02):
    """Threshold the spatial bins in `sub_ae` by number of whisks per mouse
    
    sub_ae : DataFrame
        index : MultiIndex mouse * whisker * bin_x * bin_y
        columns : ['n_whisks_per_trial', 'n_whisks']
    
    mouse_thresh : minimum number of mice to include a spatial bin
    
    nwpt_thresh : minimum mean whisks per trial to include a spatial bin
    
    Returns: MultiIndex
        names : ['bin_x', 'bin_y']
        Includes only bins to keep
    """
    # Sum n_whisks_per_trial over whiskers
    nwpt_each_mouse = sub_ae['n_whisks_per_trial'].sum(
        level=[lev for lev in sub_ae.index.names if lev != 'whisker'])

    # This should be >0 everywhere (to properly count mice below)
    assert (nwpt_each_mouse > 0).all()

    # Group nwpt_each_mouse by everything except mouse 
    # (check below that this should be bin_x and bin_y)
    grouped = nwpt_each_mouse.groupby(
        [lev for lev in nwpt_each_mouse.index.names if lev != 'mouse'])

    # Mean and size the grouped
    nwpt = grouped.mean()
    nmpb = grouped.size()

    # Error check that only bin_x and bin_y remain
    assert len(nwpt.index.names) == 2
    assert 'bin_x' in nwpt.index.names
    assert 'bin_y' in nwpt.index.names
    assert nwpt.index.equals(nmpb.index)

    # Threshold by how many whisks seem reasonable to get a good
    # estimate of the evidence
    keep_mask = (
        (nwpt >= nwpt_thresh) &
        (nmpb >= mouse_thresh)
    )
    
    # Convert to an index of 2d bins to keep
    keep_mask = keep_mask.index[keep_mask.values]
    
    return keep_mask
    
def plot_warped_edge_summary(ax, cv_ces=None, cc_ces=None, all_ces=None, 
    typ='gray'):
    """Plot warped edge summary into ax"""
    
    res = {}
    
    if typ == 'color_by_stimulus':
        res['cv_im'] = my.plot.imshow(
            cv_ces, cmap=plt.cm.Reds_r, alpha=.5, ax=ax, 
            x=cv_ces.columns.values, y=cv_ces.index.values)
        res['cc_im'] = my.plot.imshow(
            cc_ces, cmap=plt.cm.Blues_r, alpha=.5, ax=ax,
            x=cc_ces.columns.values, y=cc_ces.index.values)        
    
    elif typ == 'gray':
        res['im'] = my.plot.imshow(all_ces, cmap=plt.cm.gray_r, alpha=1, ax=ax,
            x=all_ces.columns.values, y=all_ces.index.values, clim=(0, 1))   
    
    return res

def plot_overlay(
    data, ax, full_spatial_bincenter_midx, bincenters_x, bincenters_y, 
    vmin=-1, vmax=1):
    """Plot spatial data as overlay
    
    data : Series
        index: by bin_x and bin_y
        values: intensity
    
    Returns : image handle
    """
    assert list(data.index.names) == ['bin_x', 'bin_y']
    
    # Reindex
    assert data.index.names == full_spatial_bincenter_midx.names
    topl = data.reindex(full_spatial_bincenter_midx).fillna(0)
    
    # Unstack
    topl = topl.unstack('bin_x')
    
    # Colormap
    cmap = plt.cm.RdBu_r
    
    # normalizing object
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    #~ # Plot
    #~ rgba = cmap(norm(topl.values))
    #~ rgba[:, :, 3] = 0.5
    
    # Imshow
    # Passing the norm object preserves the values for the colorbar
    im = my.plot.imshow(
        topl.values, ax=ax,
        x=bincenters_x, y=bincenters_y, 
        norm=norm, alpha=.5, cmap=cmap,
        )
    
    # Return im
    return im

