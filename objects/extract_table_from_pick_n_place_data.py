import os 
import pandas as pd
import numpy as np 
import ast
from src.helpers.helper_functions import calculate_pose_difference

class Tables:
    def __init__(self, grasp_type="PCA"): 
        self.shape_info_csv_file_path = "/Users/sophiestrawbridge/Desktop/GeometricalPickAndPlace 2/objects/objects_shape.csv"
        self.grasp_type = grasp_type
        self.output_dir = "/Users/sophiestrawbridge/Desktop/GeometricalPickAndPlace 2/results"
        os.makedirs(self.output_dir, exist_ok=True)

    def check_csv_file_exists(self, filename):
        if not os.path.exists(filename):
            print(f"The csv file {filename}")
            exit()
        df = pd.read_csv(filename)
        if df.empty:
            print("The csv file is empty")
            exit()
        return df
    
    def check_if_required_columns_are_present(self, df, required_columns):
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns in the CSV file ")
            exit()

    def analyse_trajectory_metrics(self, trajectory_attempts):
        metrics = {
            'tries': len(trajectory_attempts),
            'grasps': sum(trajectory_attempts['held_above_table_duration'] > 0),
            'collided': sum(trajectory_attempts['collided'] == True),
            'stable_grasps': sum(trajectory_attempts['stable_duration'] > 0),
            'successful_grasps': sum(trajectory_attempts['success'] > 0),
            'average_height_above_table': trajectory_attempts['max_height'].mean(), 
            'followed_trajectory': sum(trajectory_attempts['max_height'] > 0.78),
            'reached_end_of_trajectory': sum(trajectory_attempts['reached_end_of_trajectory'] == True), 
            'slipped': sum(trajectory_attempts['slipped'] == True),
        }
        pose_difference = self.calculate_average_pose_difference(trajectory_attempts)
        metrics['average_position_difference'] = pose_difference['average_position_difference']
        metrics['average_orientation_difference'] = pose_difference['average_orientation_difference']
    
        return metrics
    
    def calculate_average_pose_difference(self,df):
        if df.empty:
            return {'average_position_difference': None,
            'average_orientation_difference': None}
        object_name = df['object_name'].iloc[0]
        print(object_name)
        
        position_diffs = []
        orientation_diffs = []
        z_height_difference = []
        ids = []
        
        for idx, row in df.iterrows():
            achieved_pose_str = row['achieved_pose']
            expected_pose_str = row["expected_pose"]
            
            # Check if either pose is None
            if achieved_pose_str is None or expected_pose_str is None:
                continue
            
            # Extract position and orientation from the strings
            achieved_pose = np.array([float(x) for x in achieved_pose_str.replace("array(", "").replace(")", "").replace("[", "").replace("]", "").split(", ")])
            expected_pose = np.array([float(x) for x in expected_pose_str.replace("array(", "").replace(")", "").replace("[", "").replace("]", "").split(", ")])
            
            # Calculate position and orientation differences
            position_diff,orientation_diff = calculate_pose_difference(expected_pose[:3],achieved_pose[:3],expected_pose[3:], achieved_pose[3:])
            position_diffs.append(position_diff)
            orientation_diffs.append(orientation_diff)
            z_height_difference.append(abs(expected_pose[2] - achieved_pose[2]))
            ids.append(idx)
        
        position_df = pd.DataFrame({
            'position_diffs': position_diff,
            'orientation_diffs': orientation_diffs,
            'height': z_height_difference,
            'id':ids
        })

        output_filename = os.path.join(self.output_dir,self.grasp_type,"pose_errors",f"{object_name}_height_and_pose_differences.csv")
        position_df.to_csv(output_filename, index=False)
        
        # Calculate the average differences
        average_position_diff = np.mean(np.array(position_diffs), axis=0)
        average_orientation_diff = np.mean(np.array(orientation_diffs), axis=0)
        
        
        return {
            'average_position_difference': average_position_diff,
            'average_orientation_difference': average_orientation_diff
        }
        

    def pull_out_object_names_and_sizes_from_csv(self):
        self.object_information_df = self.check_csv_file_exists(self.shape_info_csv_file_path)
        required_columns = ['Object', 'Type', 'Length', 'Width', 'Height']
        
        self.check_if_required_columns_are_present(self.object_information_df, required_columns=required_columns)
        
        self.object_information_df[['Length', 'Width', 'Height']] = (
            self.object_information_df[['Length', 'Width', 'Height']] * 1000
        ).round().astype(int)
        
        self.filtering_object_df = pd.DataFrame({
            'Object': self.object_information_df['Object'],
            'Type': self.object_information_df['Type'],
            'Size (in mm)': (
                self.object_information_df['Length'].astype(str) + ' x ' +
                self.object_information_df['Width'].astype(str) + ' x ' +
                self.object_information_df['Height'].astype(str)
            )
        })

    def pull_out_information_from_pnp_csv(self):
        pnp_csv_filepath = os.path.join("/Users/sophiestrawbridge/Desktop/GeometricalPickAndPlace 2/data/pick_n_place_data",
                                      self.grasp_type, "pick_n_place.csv")
        
        pnp_df = self.check_csv_file_exists(pnp_csv_filepath)
        
        required_columns = ["object_name", "rank", "success", "achieved_pose", "slipped", 
                          "collided", "stable_duration", "expected_pose","achieved_pose","held_above_table_duration", 
                          "total_number_of_timesteps", "reached_end_of_trajectory"]

        self.check_if_required_columns_are_present(df=pnp_df, required_columns=required_columns)
        self.filtered_pnp_df = pnp_df
    
    def combine_pnp_data_with_shape_data(self):
        results = []
        
        for _, object_row in self.filtering_object_df.iterrows():
            object_name = object_row['Object']
            
            # Get PnP data for current object
            object_data = self.filtered_pnp_df[self.filtered_pnp_df["object_name"] == object_name]
            
            # Filter for valid trajectory attempts
            trajectory_attempts = object_data[object_data['achieved_pose'].notna()]
            
            # Get all metrics for this object
            metrics = self.analyse_trajectory_metrics(trajectory_attempts)
            
            # Calculate average rank
            valid_ranks = trajectory_attempts['rank'].dropna()
            avg_rank = valid_ranks.mean() if not valid_ranks.empty else 0
            
            # Calculate success rate
            pickup_rate = (metrics['stable_grasps'] / len(object_data)) * 100 if len(object_data) > 0 else 0
            success_rate = (metrics['successful_grasps'] / len(object_data)) * 100 if len(object_data) > 0 else 0
            # Create result row
            result = {
                'Name': object_name,
                'Type': object_row['Type'],
                'Size (in mm)': object_row['Size (in mm)'],
                'Tries': metrics['tries'],
                'Lifted of the Table': metrics['grasps'],
                'Collisions': metrics['collided'],
                'Slips':metrics['slipped'],
                "Picked up above 1cm": metrics['stable_grasps'],
                'Successful stable grasps':  metrics['successful_grasps'],
                'Average Rank': round(avg_rank, 2) if avg_rank != 0 else 0,
                'Pick up Rate (%)': round(pickup_rate, 1),
                'Stable Grasp (%)': round(success_rate, 1),
                'Followed trajectory': metrics['followed_trajectory'],
                'Reached end of trajectory': metrics['reached_end_of_trajectory'],
                'average orientation difference': metrics['average_orientation_difference'],
                'average position difference':metrics['average_position_difference']
            }
            
            results.append(result)
        
        # Create and sort DataFrame
        self.combined_df = pd.DataFrame(results)
        self.combined_df = self.combined_df.sort_values(['Type', 'Stable Grasp (%)'], ascending=[True, False])
        
        # Save to CSV
        output_filename = os.path.join(self.output_dir, f'pnp_results_{self.grasp_type}.csv')
        self.combined_df.to_csv(output_filename, index=False)
        print(f"\nResults saved to: {output_filename}")
        
        return self.combined_df
    
extract_tables = Tables(grasp_type="PCA")
extract_tables.pull_out_object_names_and_sizes_from_csv()
print(extract_tables.pull_out_information_from_pnp_csv())
print(extract_tables.combine_pnp_data_with_shape_data())