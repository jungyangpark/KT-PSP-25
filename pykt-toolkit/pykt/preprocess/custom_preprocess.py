import pandas as pd
import numpy as np
try:
    from .utils import sta_infos, write_txt, format_list2str
except ImportError:
    from pykt.preprocess.utils import sta_infos, write_txt, format_list2str

KEYS = ["uid", "problem_id", "concepts"]

def read_data_from_csv(read_file, write_file):
    """
    Process custom data format from final_data.csv and convert to pyKT data.txt format.
    
    Expected input format:
    - fold: cross-validation fold
    - uid: user ID
    - problem_id: comma-separated problem IDs  
    - concepts: comma-separated concept IDs
    - responses: comma-separated responses (0/1)
    - timestamp: comma-separated timestamps
    - response_time: comma-separated response times
    - error_type: comma-separated error types (5 values per problem, currently flattened)
    - mathematical_proficiency: comma-separated error types (8 values per problem, currently flattened)

    pyKT data.txt format (19 lines per student):
    Line 0: user_id,sequence_length
    Line 1: question_ids (comma-separated)
    Line 2: concept_ids (comma-separated)
    Line 3: responses (comma-separated)
    Line 4: timestamps (comma-separated)
    Line 5: response_times (comma-separated)
    Line 6: error_type_0 (comma-separated)
    Line 7: error_type_1 (comma-separated)
    Line 8: error_type_2 (comma-separated)
    Line 9: error_type_3 (comma-separated)
    Line 10: error_type_4 (comma-separated)
    Line 11: math_prof_0 (comma-separated)
    Line 12: math_prof_1 (comma-separated)
    Line 13: math_prof_2 (comma-separated)
    Line 14: math_prof_3 (comma-separated)
    Line 15: math_prof_4 (comma-separated)
    Line 16: math_prof_5 (comma-separated)
    Line 17: math_prof_6 (comma-separated)
    Line 18: math_prof_7 (comma-separated)

    """

    print(f"Processing custom data from: {read_file}")
    
    # Read CSV data
    df = pd.read_csv(read_file, dtype=str)
    
    stares = []
    
    # Calculate initial statistics
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # Drop rows with missing essential data
    df['tmp_index'] = range(len(df))
    _df = df.dropna(subset=["uid", "problem_id", "concepts", "responses"])
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # Group by user
    ui_df = _df.groupby('uid', sort=False)
    
    user_inters = []
    
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        
        # Sort by timestamp if available
        if 'timestamp' in tmp_inter.columns and not tmp_inter['timestamp'].iloc[0] == 'NA':
            # Convert timestamp string to list and sort
            timestamps = tmp_inter['timestamp'].iloc[0].split(',')
            if len(timestamps) > 1:
                # Create sorting indices based on timestamp
                timestamp_indices = sorted(range(len(timestamps)), key=lambda i: int(timestamps[i]))
            else:
                timestamp_indices = list(range(len(tmp_inter)))
        else:
            timestamp_indices = list(range(len(tmp_inter)))
        
        # Process each user's data
        for _, row in tmp_inter.iterrows():
            # Parse comma-separated values
            problem_ids = row['problem_id'].split(',')
            concept_ids = row['concepts'].split(',')
            responses = row['responses'].split(',')
            
            # Handle timestamps
            if pd.notna(row['timestamp']) and row['timestamp'] != 'NA':
                timestamps = row['timestamp'].split(',')
            else:
                timestamps = [str(i) for i in range(len(problem_ids))]
            
            # Handle response times
            if pd.notna(row.get('response_time', 'NA')) and row['response_time'] != 'NA':
                response_times = row['response_time'].split(',')
            else:
                response_times = ['NA'] * len(problem_ids)
            
            # Handle error types (5 values per problem, stored as 5 separate arrays)
            error_types_5d = [[], [], [], [], []]  # 5 separate arrays for each error type
            if pd.notna(row.get('error_type', 'NA')) and row['error_type'] != 'NA':
                error_type_values = row['error_type'].split(',')
                # If we have 5x the number of problems, it means each problem has 5 error type values
                if len(error_type_values) == len(problem_ids) * 5:
                    # Group every 5 values into separate arrays
                    for i in range(len(problem_ids)):
                        start_idx = i * 5
                        for j in range(5):
                            try:
                                error_val = int(error_type_values[start_idx + j])
                            except (ValueError, IndexError):
                                error_val = 0
                            error_types_5d[j].append(str(error_val))
                else:
                    # Fill with default values
                    for j in range(5):
                        error_types_5d[j] = ['0'] * len(problem_ids)
            else:
                # No error type data, fill with zeros
                for j in range(5):
                    error_types_5d[j] = ['0'] * len(problem_ids)

            # Handle mathematical proficiency (8 values per problem, stored as 8 separate arrays)
            math_prof_8d = [[], [], [], [], [], [], [], []]  # 8 separate arrays for each math proficiency type
            if pd.notna(row.get('mathematical_proficiency', 'NA')) and row['mathematical_proficiency'] != 'NA':
                math_prof_values = row['mathematical_proficiency'].split(',')
                # If we have 8x the number of problems, it means each problem has 8 math proficiency values
                if len(math_prof_values) == len(problem_ids) * 8:
                    # Group every 8 values into separate arrays
                    for i in range(len(problem_ids)):
                        start_idx = i * 8
                        for j in range(8):
                            try:
                                math_prof_val = int(math_prof_values[start_idx + j])
                            except (ValueError, IndexError):
                                math_prof_val = 0
                            math_prof_8d[j].append(str(math_prof_val))
                else:
                    # Fill with default values
                    for j in range(8):
                        math_prof_8d[j] = ['0'] * len(problem_ids)
            else:
                # No mathematical proficiency data, fill with zeros
                for j in range(8):
                    math_prof_8d[j] = ['0'] * len(problem_ids)
            
            seq_len = len(problem_ids)
            
            # Ensure all sequences have the same length
            assert seq_len == len(concept_ids) == len(responses), f"Sequence length mismatch for user {user}"
            
            # Truncate or pad other sequences to match problem_ids length
            timestamps = timestamps[:seq_len] + ['NA'] * max(0, seq_len - len(timestamps))
            response_times = response_times[:seq_len] + ['NA'] * max(0, seq_len - len(response_times))
            
            # Ensure error type arrays have correct length
            for j in range(5):
                if len(error_types_5d[j]) < seq_len:
                    error_types_5d[j].extend(['0'] * (seq_len - len(error_types_5d[j])))
                elif len(error_types_5d[j]) > seq_len:
                    error_types_5d[j] = error_types_5d[j][:seq_len]

            # Ensure mathematical proficiency arrays have correct length
            for j in range(8):
                if len(math_prof_8d[j]) < seq_len:
                    math_prof_8d[j].extend(['0'] * (seq_len - len(math_prof_8d[j])))
                elif len(math_prof_8d[j]) > seq_len:
                    math_prof_8d[j] = math_prof_8d[j][:seq_len]
            
            user_inters.append([
                [str(user), str(seq_len)],
                format_list2str(problem_ids),
                format_list2str(concept_ids),
                format_list2str(responses),
                format_list2str(timestamps),
                format_list2str(response_times),
                format_list2str(error_types_5d[0]),  # error_type_0
                format_list2str(error_types_5d[1]),  # error_type_1
                format_list2str(error_types_5d[2]),  # error_type_2
                format_list2str(error_types_5d[3]),  # error_type_3
                format_list2str(error_types_5d[4]),  # error_type_4
                format_list2str(math_prof_8d[0]),    # math_prof_0
                format_list2str(math_prof_8d[1]),    # math_prof_1
                format_list2str(math_prof_8d[2]),    # math_prof_2
                format_list2str(math_prof_8d[3]),    # math_prof_3
                format_list2str(math_prof_8d[4]),    # math_prof_4
                format_list2str(math_prof_8d[5]),    # math_prof_5
                format_list2str(math_prof_8d[6]),    # math_prof_6
                format_list2str(math_prof_8d[7])     # math_prof_7
            ])
    
    # Write to output file
    write_txt(write_file, user_inters)
    
    print("\n".join(stares))
    print(f"Processed {len(user_inters)} user sequences")
    print(f"Written to: {write_file}")
    
    return write_file