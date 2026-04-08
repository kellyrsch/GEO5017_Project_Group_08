import pandas as pd

from data import get_image_fps, get_dino_labelling_data
import shutil
from pathlib import Path


def create_manual_labelling_subset(
        total_samples: int = 500,
        no_waste_samples: int = 200
    ):
    labels = get_dino_labelling_data()
    all_images = get_image_fps()
    image_metadata_lookup = {fp: (year, heading, pitch) for fp, year, heading, pitch in all_images}

    # Merge scores with metadata
    df = pd.DataFrame(labels, columns=['filename', 'max_confidence', 'waste_objects_detected'])
    df['year'] = df['filename'].apply(lambda x: image_metadata_lookup[x][0])
    df['heading'] = df['filename'].apply(lambda x: image_metadata_lookup[x][1])
    df['pitch'] = df['filename'].apply(lambda x: image_metadata_lookup[x][2])
    
    waste_sample_size = total_samples - no_waste_samples
    
    # Calculate the "Ground Truth" Year Distribution from the FULL dataset
    year_distribution = df['year'].value_counts(normalize=True)
    
    selected_waste_images = []
    selected_clean_images = []
    
    # Process year by year to strictly enforce stratification
    for year, proportion in year_distribution.items():
        year_df = df[df['year'] == year]
        
        y_waste_quota = int(round(waste_sample_size * proportion))
        y_clean_quota = int(round(no_waste_samples * proportion))
        
        potential_waste = year_df[year_df['max_confidence'] > 0].sort_values(by='max_confidence', ascending=False)
        
        if len(potential_waste) > y_waste_quota:
            # To maintain heading/pitch diversity, we don't just take the top N.
            # We take the top 3x candidates, then randomly sample the quota from them.
            # This guarantees high confidence AND respects the natural variance of the camera angles.
            top_k_pool = potential_waste.head(y_waste_quota * 3)
            sampled_waste = top_k_pool.sample(n=y_waste_quota, random_state=42)
        else:
            # If a year doesn't have enough waste candidates, take whatever is available
            sampled_waste = potential_waste
            
        selected_waste_images.append(sampled_waste)
        
        potential_clean = year_df[year_df['max_confidence'] == 0]
        
        if len(potential_clean) > y_clean_quota:
            sampled_clean = potential_clean.sample(n=y_clean_quota, random_state=42)
        else:
            sampled_clean = potential_clean
            
        selected_clean_images.append(sampled_clean)

    final_waste_df = pd.concat(selected_waste_images)
    final_clean_df = pd.concat(selected_clean_images)
    final_subset = pd.concat([final_waste_df, final_clean_df])
    
    # Handle minor rounding errors (e.g., if rounding yields 999 instead of 1000)
    current_total = len(final_subset)
    if current_total > total_samples:
        # Too many? Drop a few random ones
        final_subset = final_subset.sample(n=total_samples, random_state=42)
    
    print(f"Successfully generated subset of {len(final_subset)} images.")
    print(f"Waste Images: {len(final_waste_df)} | Clean Images: {len(final_clean_df)}")
    print("\nYear Distribution in Subset compared to Original:")
    
    # Verification printout
    subset_dist = final_subset['year'].value_counts(normalize=True).sort_index()
    orig_dist = year_distribution.sort_index()
    for y in orig_dist.index:
        print(f"{y}: Original {orig_dist[y]*100:.1f}% -> Subset {subset_dist.get(y, 0)*100:.1f}%")

    # copy all files in the final subset to a new folder for manual labelling
    # Create output directory
    output_dir = Path("data/manual_labelling_subset")
    output_dir.mkdir(exist_ok=True)

    # Copy files
    for filename in final_subset['filename']:
        src = Path(filename)
        if src.exists():
            shutil.copy2(src, output_dir / src.name)
    return final_subset


if __name__ == "__main__":
    create_manual_labelling_subset()