from pathlib import Path

import pandas as pd


def unique_ids_from_sequence_column(series: pd.Series):
    values = set()
    for entry in series:
        if isinstance(entry, list):
            values.update(int(item) for item in entry)
        elif hasattr(entry, 'tolist'):
            values.update(int(item) for item in entry.tolist())
    return values


def summarize_membership(name: str, ids: set[int], caption_ids: set[int], sid_ids: set[int]):
    print(f'\n=== {name}')
    print('count=', len(ids))
    print('caption_covered=', len(ids & caption_ids))
    print('caption_missing=', len(ids - caption_ids))
    print('sid_covered=', len(ids & sid_ids))
    print('sid_missing=', len(ids - sid_ids))


def main():
    root = Path.cwd()
    main_df = pd.read_parquet(root / 'onerec_bench_release.parquet', columns=[
        'hist_video_pid', 'target_video_pid',
        'hist_ad_pid', 'target_ad_pid',
        'hist_goods_pid', 'target_goods_pid',
    ])
    caption_df = pd.read_parquet(root / 'pid2caption.parquet')
    product_sid_df = pd.read_parquet(root / 'product_pid2sid.parquet')
    video_ad_sid_df = pd.read_parquet(root / 'video_ad_pid2sid.parquet')

    video_ids = unique_ids_from_sequence_column(main_df['hist_video_pid']) | unique_ids_from_sequence_column(main_df['target_video_pid'])
    ad_ids = unique_ids_from_sequence_column(main_df['hist_ad_pid']) | unique_ids_from_sequence_column(main_df['target_ad_pid'])
    goods_ids = unique_ids_from_sequence_column(main_df['hist_goods_pid']) | unique_ids_from_sequence_column(main_df['target_goods_pid'])
    video_ad_ids = video_ids | ad_ids

    caption_ids = set(int(pid) for pid in caption_df['pid'].tolist())
    product_sid_ids = set(int(pid) for pid in product_sid_df['pid'].tolist())
    video_ad_sid_ids = set(int(pid) for pid in video_ad_sid_df['pid'].tolist())

    print('ROOT=', root)
    print('\n=== BASIC COUNTS')
    print('caption_pid_count=', len(caption_ids))
    print('product_sid_pid_count=', len(product_sid_ids))
    print('video_ad_sid_pid_count=', len(video_ad_sid_ids))

    summarize_membership('video', video_ids, caption_ids, video_ad_sid_ids)
    summarize_membership('ad', ad_ids, caption_ids, video_ad_sid_ids)
    summarize_membership('goods', goods_ids, caption_ids, product_sid_ids)

    print('\n=== DOMAIN OVERLAP')
    print('video_ad_overlap=', len(video_ids & ad_ids))
    print('video_goods_overlap=', len(video_ids & goods_ids))
    print('ad_goods_overlap=', len(ad_ids & goods_ids))
    print('video_ad_goods_overlap=', len(video_ids & ad_ids & goods_ids))

    print('\n=== SID TABLE OVERLAP')
    sid_pid_overlap = product_sid_ids & video_ad_sid_ids
    print('product_vs_video_ad_pid_overlap=', len(sid_pid_overlap))
    sample_overlap = sorted(list(sid_pid_overlap))[:20]
    print('sample_overlap_pids=', sample_overlap)

    if sample_overlap:
        product_map = product_sid_df.set_index('pid')['sid'].to_dict()
        video_ad_map = video_ad_sid_df.set_index('pid')['sid'].to_dict()
        print('\n=== SAMPLE OVERLAP SID COMPARISON')
        for pid in sample_overlap[:10]:
            print(
                {
                    'pid': int(pid),
                    'product_sid': product_map.get(pid),
                    'video_ad_sid': video_ad_map.get(pid),
                    'same_sid': product_map.get(pid) == video_ad_map.get(pid),
                }
            )

    print('\n=== CROSS-COVERAGE CHECK')
    print('goods_ids_found_in_video_ad_sid_table=', len(goods_ids & video_ad_sid_ids))
    print('video_ad_ids_found_in_product_sid_table=', len(video_ad_ids & product_sid_ids))

    print('\n=== UNIFIED PID FEASIBILITY (STRICT)')
    problematic_goods = (goods_ids & video_ad_ids) | (goods_ids & video_ad_sid_ids)
    print('goods_conflicting_with_video_or_ad_space=', len(problematic_goods))
    print('sample_problematic_goods=', sorted(list(problematic_goods))[:20])


if __name__ == '__main__':
    main()
