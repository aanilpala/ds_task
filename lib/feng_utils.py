def calc_ttype_share_features(group, ttypes):
    volume_by_ttype = group.groupby("transaction_type", dropna=True).agg(
        {"amount_internal_currency": ["sum"]}
    )["amount_internal_currency"]["sum"]
    total_volume = volume_by_ttype.sum()
    share_by_ttype = volume_by_ttype / total_volume

    share_by_ttype.rename(lambda val: f"ttype_{val}_share", inplace=True)

    ttype_share_features_dict = share_by_ttype.to_dict()
    ttype_share_features_dict["num_ttypes_seen"] = len(
        ttype_share_features_dict.values()
    )

    for ttype in ttypes:
        if not ttype_share_features_dict.get(f"ttype_{ttype}_share"):
            ttype_share_features_dict[f"ttype_{ttype}_share"] = 0.0

    # Not necessary as ttype is never null
    # if not ttype_features_dict.get(f'ttype_nan_share'):
    #     ttype_features_dict[f'ttype_nan_share'] = 0.0

    return ttype_share_features_dict


def calc_mccgroup_share_features(group, mcc_groups):
    volume_by_mcc_group = group.groupby("mcc_group", dropna=False).agg(
        {"amount_internal_currency": ["sum"]}
    )["amount_internal_currency"]["sum"]
    total_volume = volume_by_mcc_group.sum()
    share_by_mcc_group = volume_by_mcc_group / total_volume

    # val != val means null
    share_by_mcc_group.rename(
        lambda val: f"mcc_{int(val)}_share" if val == val else "mcc_nan_share",
        inplace=True,
    )

    mcc_group_share_features_dict = share_by_mcc_group.to_dict()
    mcc_group_share_features_dict["num_mcc_groups_seen"] = len(
        mcc_group_share_features_dict.values()
    )

    for mcc_group in mcc_groups:
        if not mcc_group_share_features_dict.get(f"mcc_{mcc_group}_share"):
            mcc_group_share_features_dict[f"mcc_{mcc_group}_share"] = 0.0

    if not mcc_group_share_features_dict.get(f"mcc_nan_share"):
        mcc_group_share_features_dict[f"mcc_nan_share"] = 0.0

    return mcc_group_share_features_dict


def calc_ttype_features(ttype_series, ttypes):
    ttype_counts = ttype_series.value_counts()

    ttype_counts.rename(lambda val: f"ttype_{val}_count", inplace=True)

    ttype_features_dict = ttype_counts.to_dict()
    ttype_features_dict["num_ttypes_seen"] = len(ttype_features_dict.values())

    for ttype in ttypes:
        if not ttype_features_dict.get(f"ttype_{ttype}_count"):
            ttype_features_dict[f"ttype_{ttype}_count"] = 0

    return ttype_features_dict


def calc_mccgroup_features(mcc_group_series, mcc_groups):
    mcc_group_counts = mcc_group_series.value_counts(dropna=False)

    # val != val means null
    mcc_group_counts.rename(
        lambda val: f"mcc_{int(val)}_count" if val == val else "mcc_nan_count",
        inplace=True,
    )

    mcc_group_features_dict = mcc_group_counts.to_dict()
    mcc_group_features_dict["num_mcc_groups_seen"] = len(
        mcc_group_features_dict.values()
    )

    for mcc_group in mcc_groups:
        if not mcc_group_features_dict.get(f"mcc_{int(mcc_group)}_count"):
            mcc_group_features_dict[f"mcc_{int(mcc_group)}_count"] = 0

    if not mcc_group_features_dict.get(f"mcc_nan_count"):
        mcc_group_features_dict[f"mcc_nan_count"] = 0

    return mcc_group_features_dict


def calc_features(group, ttypes, mcc_groups):

    ttype_feats = calc_ttype_features(group["transaction_type"], ttypes)
    ttype_share_feats = calc_ttype_share_features(group, ttypes)
    mccgroup_feats = calc_mccgroup_features(group["mcc_group"], mcc_groups)
    mccgroup_share_feats = calc_mccgroup_share_features(group, mcc_groups)

    return {
        **ttype_feats,
        **ttype_share_feats,
        **mccgroup_feats,
        **mccgroup_share_feats,
    }
