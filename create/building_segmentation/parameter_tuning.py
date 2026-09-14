import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate import train, validate
from tiling import generateTiles, generateDataYAML, loadGroundTruthBySource
from config import (
    getFinalModelConfig,
    SEED,
    VALIDATING_IMAGES_DIR,
    VALIDATING_DETECTION_FILE,
    TRAINING_IMAGES_DIR,
    TRAINING_DETECTION_FILE,
    TRAINING_TILES_DIR,
    TRAINING_TILE_INDEX_FILE,
    TRAINING_DATASET_DIR,
    TRAINING_DATASET_FILE,
    VALIDATING_TILES_DIR,
    VALIDATING_TILE_INDEX_FILE,
    OUTPUT_MODELS_DIR,
)

TILE_SIZES = [320, 512, 640, 960]
OVERLAPS = [0, 32, 64, 128, 192]
MODEL_TYPES = ['YOLO', 'MASK-RCNN']
IOU_THRESHS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def plotHeatmapPerModelTileSizeOverlap(tuning_results, column_name: str, title: str):
    tile_sizes = sorted(tuning_results['tile_size'].unique())
    overlaps = sorted(tuning_results['overlap'].unique())
    models = sorted(tuning_results['model'].unique())

    cmap = plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for idx, model in enumerate(models):
        data = tuning_results[tuning_results['model'] == model]
        heatmap_data = data.pivot_table(index='tile_size', columns='overlap', values=column_name, aggfunc='mean')
        heatmap_data = heatmap_data.reindex(index=tile_sizes, columns=overlaps)
        heatmap_values = heatmap_data.values
        ax = axes[idx]
        im = ax.imshow(heatmap_values, aspect="auto", cmap=cmap, vmin=np.nanmin(heatmap_values), vmax=np.nanmax(heatmap_values))
        ax.set_xticks(np.arange(len(overlaps)))
        ax.set_yticks(np.arange(len(tile_sizes)))
        ax.set_xticklabels(overlaps)
        ax.set_yticklabels(tile_sizes if idx == 1 else [""]*len(tile_sizes))
        ax.set_xlabel('Overlap')
        if idx == 0:
            ax.set_ylabel('Tile Size')
        
        for i in range(len(tile_sizes)):
            for j in range(len(overlaps)):
                val = heatmap_values[i, j]
                text = f"{val:.2f}" if not np.isnan(val) else "NA"
                ax.text(j, i, text, ha="center", va="center", color="black" if np.isnan(val) or val < (np.nanmax(heatmap_values) * 0.7) else "white", fontsize=10)
        
        ax.set_title(model)
        if idx == 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=title)

    plt.suptitle(f"{title} Heatmap")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'output/parameter_tuning/{column_name}_per_model_per_tilesize_per_overlap.jpg', dpi=300)
    plt.close()

def plotHeatmapPerModelEnv(tuning_results, column_name: str, title: str):
    envs = sorted(tuning_results['env'].unique())
    tile_sizes = sorted(tuning_results['tile_size'].unique())
    overlaps = sorted(tuning_results['overlap'].unique())
    models = sorted(tuning_results['model'].unique())

    cmap = plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    for idx_c, model in enumerate(models):
        for idx_r, param in enumerate([tile_sizes, overlaps]):
            ax = axes[idx_r, idx_c]
            data = tuning_results[tuning_results['model'] == model]
            if idx_r == 0:
                heatmap_data = data.pivot_table(index='env', columns='tile_size', values=column_name, aggfunc='mean')
                ax.set_xlabel('Tile Size')
            else:
                heatmap_data = data.pivot_table(index='env', columns='overlap', values=column_name, aggfunc='mean')
                ax.set_xlabel('Overlap')
                
            heatmap_data = heatmap_data.reindex(index=envs, columns=param)
            heatmap_values = heatmap_data.values
            im = ax.imshow(heatmap_values, aspect="auto", cmap=cmap, vmin=np.nanmin(heatmap_values), vmax=np.nanmax(heatmap_values))
            ax.set_xticks(np.arange(len(param)))
            ax.set_yticks(np.arange(len(envs)))
            ax.set_xticklabels(param)
            ax.set_yticklabels(envs if idx_c == 1 else [""]*len(envs))
            if idx_c == 0:
                ax.set_ylabel('Environment')
            
            for i in range(len(envs)):
                for j in range(len(param)):
                    val = heatmap_values[i, j]
                    text = f"{val:.2f}" if not np.isnan(val) else "NA"
                    ax.text(j, i, text, ha="center", va="center", color="black" if np.isnan(val) or val < (np.nanmax(heatmap_values) * 0.7) else "white", fontsize=10)
            
            ax.set_title(model)
            if idx_c == 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=title)
    
    plt.suptitle(f"{title} Heatmap")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'output/parameter_tuning/{column_name}_per_model_per_env.jpg', dpi=300)
    plt.close()

def drawComparisonChart(file_path):
    envs = {
        'crop_0': 'mixed',
        'crop_1': 'tall',
        'crop_2': 'wide',
        'crop_3': 'residential',
    }

    tuning_results = pd.read_csv(file_path, index_col=0)
    
    tuning_results['env'] = None
    for ind, row in tuning_results.iterrows():
        for env, type in envs.items():
            if env in row['source']:
                tuning_results.at[ind, 'env'] = type

    tuning_results['f1'] = 2 * (tuning_results['precision'] * tuning_results['recall']) / (tuning_results['precision'] + tuning_results['recall'])
    tuning_results['f1'] = tuning_results['f1'].fillna(0)
    tuning_results['f1_weighted'] = tuning_results['f1'] * tuning_results['actual']

    plotHeatmapPerModelTileSizeOverlap(tuning_results, 'f1', 'F1-Score')
    plotHeatmapPerModelTileSizeOverlap(tuning_results, 'f1_weighted', 'Weighted F1-Score')
    plotHeatmapPerModelTileSizeOverlap(tuning_results, 'dice', 'Dice')
    plotHeatmapPerModelTileSizeOverlap(tuning_results, 'iou', 'IoU')
    plotHeatmapPerModelTileSizeOverlap(tuning_results, 'validation_duration_sec', 'Inference Duration (sec)')
    plotHeatmapPerModelEnv(tuning_results, 'f1', 'F1-Score')
    plotHeatmapPerModelEnv(tuning_results, 'dice', 'Dice')
    plotHeatmapPerModelEnv(tuning_results, 'validation_duration_sec', 'Inference Duration (sec)')

def main():
    os.makedirs('output/parameter_tuning/', exist_ok=True)
    config = getFinalModelConfig()
    truth_by_source = loadGroundTruthBySource(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE)
    
    results = []
    for tile_size in TILE_SIZES:
        for overlap in OVERLAPS:
            training_tile_index = generateTiles(TRAINING_IMAGES_DIR, TRAINING_DETECTION_FILE, TRAINING_TILES_DIR, TRAINING_TILE_INDEX_FILE, tile_size, overlap, gsd_m=config['gsd_minimum'])
            training_data_file = generateDataYAML(training_tile_index, TRAINING_TILES_DIR, TRAINING_DATASET_DIR, TRAINING_DATASET_FILE, 0.2, SEED)
            validation_tile_index = generateTiles(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE, tile_size, overlap, gsd_m=config['gsd_minimum'])
            for model_type in MODEL_TYPES:
                model = train(model_type, config[model_type], training_tile_index, TRAINING_TILES_DIR, training_data_file, tile_size, OUTPUT_MODELS_DIR)

                t0 = time.time()
                model_performance = validate(model_type, model, config[model_type], VALIDATING_TILES_DIR, validation_tile_index, tile_size, truth_by_source, config['nms_iou_threshold'], config['accuracy_iou_threshold'])
                duration = time.time() - t0

                for validation_row in model_performance:
                    results.append({
                        'tile_size': tile_size, 
                        'overlap': overlap, 
                        'model': model_type, 
                        'validation_duration_sec': round(duration, 2),
                        'source': validation_row['source'], 
                        'actual': int(validation_row['actual']), 
                        'predicted': int(validation_row['predicted']), 
                        'precision': round(validation_row['precision'], 4), 
                        'recall': round(validation_row['recall'], 4), 
                        'iou': round(validation_row['iou'], 4), 
                        'dice': round(validation_row['dice'], 4)
                    })
               
                results_df = pd.DataFrame(results)
                results_df.to_csv('output/parameter_tuning/parameter_tuning.csv')
    
    drawComparisonChart('output/parameter_tuning/parameter_tuning.csv')

if __name__ == '__main__':
    main()
