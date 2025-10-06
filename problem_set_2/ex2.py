import random
import time
import multiprocessing
import matplotlib.pyplot as plt
random.seed(0)

def alg2(keys, address):
    if len(keys) <= 1:
        return keys, address
    else:
        split = len(keys) // 2
        left, left_address = alg2(keys[:split],address[:split])
        right, right_address = alg2(keys[split:],address[split:])
        left = iter(left)
        right = iter(right)
        left_address = iter(left_address)
        right_address = iter(right_address)
        result = []
        result_address = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        left_addr_top = next(left_address)
        right_addr_top = next(right_address)
        while True:
            if left_top < right_top:
                result.append(left_top)
                result_address.append(left_addr_top)
                try:
                    left_top = next(left)
                    left_addr_top = next(left_address)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right), result_address + [right_addr_top] + list(right_address)
            else:
                result.append(right_top)
                result_address.append(right_addr_top)
                try:
                    right_top = next(right)
                    right_addr_top = next(right_address)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left), result_address + [left_addr_top] + list(left_address)
                
def alg2_dict(data, key):
    # print(f'Original {key}s: {[d[key] for d in data]}')
    keys = [d[key] for d in data]
    addr = list(range(len(data)))
    keys_sorted, addr_sorted = alg2(keys, addr)
    # print(f'Sorted {key}s: {keys_sorted}')
    return [data[i] for i in addr_sorted]

def alg2_parallel(keys, address, pool): 
    if len(keys) <= 1:
        return keys, address
    else:
        split = len(keys) // 2
        # Parallel processing
        left_task = pool.apply_async(alg2, (keys[:split], address[:split]))
        right_task = pool.apply_async(alg2, (keys[split:], address[split:]))
        
        left, left_address = left_task.get()
        right, right_address = right_task.get()

        left = iter(left)
        right = iter(right)
        left_address = iter(left_address)
        right_address = iter(right_address)
        result = []
        result_address = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        left_addr_top = next(left_address)
        right_addr_top = next(right_address)
        while True:
            if left_top < right_top:
                result.append(left_top)
                result_address.append(left_addr_top)
                try:
                    left_top = next(left)
                    left_addr_top = next(left_address)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right), result_address + [right_addr_top] + list(right_address)
            else:
                result.append(right_top)
                result_address.append(right_addr_top)
                try:
                    right_top = next(right)
                    right_addr_top = next(right_address)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left), result_address + [left_addr_top] + list(left_address)

def alg2_dict_parallel(data, key, pool):
    # print(f'Original {key}s: {[d[key] for d in data]}')
    keys = [d[key] for d in data]
    addr = list(range(len(data)))
    _, addr_sorted = alg2_parallel(keys, addr, pool)
    # keys_sorted, addr_sorted = alg2_parallel(keys, addr, pool)
    # print(f'Sorted {key}s: {keys_sorted}')
    return [data[i] for i in addr_sorted]

def merge_sorted_lists(list1, list2, key):
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i][key] < list2[j][key]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

def alg2_dict_chunked(data, key, pool, num_chunks):
    chunk_size = len(data) // num_chunks
    tasks = []
    
    for i in range(num_chunks):
        chunk = data[i*chunk_size : (i+1)*chunk_size] if i < num_chunks - 1 else data[i*chunk_size :]
        tasks.append(pool.apply_async(alg2_dict, (chunk, key)))
    sorted_chunks = [task.get() for task in tasks]

    # Merge sorted chunks
    while len(sorted_chunks) > 1:
        merged_chunks = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                merged_chunks.append(merge_sorted_lists(sorted_chunks[i], sorted_chunks[i + 1], key))
            else:
                merged_chunks.append(sorted_chunks[i])
        sorted_chunks = merged_chunks

    return sorted_chunks[0]

# def merge_sorted_lists_opt(keys1, keys2, addrs1, addrs2):
#     merged_keys = []
#     merged_addrs = []
#     i, j = 0, 0
#     while i < len(keys1) and j < len(keys2):
#         if keys1[i] < keys2[j]:
#             merged_keys.append(keys1[i])
#             merged_addrs.append(addrs1[i])
#             i += 1
#         else:
#             merged_keys.append(keys2[j])
#             merged_addrs.append(addrs2[j])
#             j += 1
#     merged_keys.extend(keys1[i:])
#     merged_keys.extend(keys2[j:])
#     merged_addrs.extend(addrs1[i:])
#     merged_addrs.extend(addrs2[j:])
#     return merged_keys, merged_addrs

# def alg2_dict_chunked_opt(data, key, pool, num_chunks):
#     chunk_size = len(data) // num_chunks
#     keys = [d[key] for d in data]
#     addr = list(range(len(data)))
#     sorted_keys = []
#     sorted_addrs = []
    
#     for i in range(num_chunks):
#         key_chunk = keys[i*chunk_size : (i+1)*chunk_size] if i < num_chunks - 1 else keys[i*chunk_size :]
#         addr_chun = addr[i*chunk_size : (i+1)*chunk_size] if i < num_chunks - 1 else addr[i*chunk_size :]
#         sorted_key, sorted_addr = (pool.apply_async(alg2, (key_chunk, addr_chun))).get()
#         sorted_keys.append(sorted_key)
#         sorted_addrs.append(sorted_addr)

#     # Merge sorted chunks
#     while len(sorted_keys) > 1:
#         merged_keys = []
#         merged_addrs = []
#         for i in range(0, len(sorted_keys), 2):
#             if i + 1 < len(sorted_keys):
#                 new_mer_keys, new_mer_addrs = merge_sorted_lists_opt(sorted_keys[i], sorted_keys[i+1], sorted_addrs[i], sorted_addrs[i+1])
#                 merged_keys.append(new_mer_keys)
#                 merged_addrs.append(new_mer_addrs)
#             else:
#                 merged_addrs.append(sorted_addrs[i])
#                 merged_keys.append(sorted_keys[i])
#         sorted_keys = merged_keys
#         sorted_addrs = merged_addrs

#     return [data[i] for i in sorted_addrs[0]]

if __name__ == "__main__":

    # -----2a. Validation-----
    # n = 5
    # data = [{'patient_id': i, 'patient_data': chr(97 + i), 'age': random.randint(1, 99)} for i in random.sample(range(n), k=n)]

    # print("Unsorted data:")
    # print(data)
    # print("\nSort by patient_id:")
    # print(alg2_dict(data, 'patient_id'))
    # print("\nSort by age:")
    # print(alg2_dict(data, 'age'))

    # -----2b. Parallelize the Algorithm-----
    n = 10000000
    ns = [int(10**i) for i in range(1, 8)]
    # ns = [n]
    # n = 100000
    chunked_times = []
    parallel_times = []
    serial_times = []
    for n in ns:
        data = [{'patient_id': i, 'patient_data': chr(random.randint(97,123)), 'age': random.randint(1, 99)} for i in random.sample(range(n), k=n)]
        rep_time = 5
        num_cores = multiprocessing.cpu_count()
        # print(f'Number of CPU cores: {num_cores}')
        shortest_time = 999999
        for i in range(rep_time):
            with multiprocessing.Pool(processes=num_cores) as pool:
                start_time = time.perf_counter()
                # Divide data into chunks for each core
                alg2_dict_chunked(data, 'patient_id', pool, num_cores)
                end_time = time.perf_counter()
            shortest_time = min(shortest_time, end_time - start_time)
        chunked_times.append(shortest_time)
        print(f'Elapsed time (parallel): {shortest_time} seconds')

        # shortest_time = 999999
        # for i in range(rep_time):
        #     with multiprocessing.Pool(processes=num_cores) as pool:
        #         start_time = time.perf_counter()
        #         alg2_dict_parallel(data, 'patient_id', pool)
        #         end_time = time.perf_counter()
        #     shortest_time = min(shortest_time, end_time - start_time)
        # # print(f'Elapsed time (parallel): {shortest_time} seconds')
        # parallel_times.append(shortest_time)

        # Serial version
        shortest_time = 999999
        for i in range(rep_time):
            start_time = time.perf_counter()
            alg2_dict(data, 'patient_id')
            end_time = time.perf_counter()
            shortest_time = min(shortest_time, end_time - start_time)
        serial_times.append(shortest_time)
        print(f'Elapsed time (serial): {shortest_time} seconds')
    
    plt.figure(figsize=(10, 7))
    plt.plot(ns, serial_times, marker='.', label='Serial')
    plt.plot(ns, chunked_times, marker='.', label='Parallel')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Algorithm Performance vs. Dataset Size')
    plt.xlabel('Dataset Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("algorithm_performance_plot.png")