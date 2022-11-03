import multiprocessing
# import tqdm
# from tqdm.auto import  tqdm as tq
import tqdm.auto as tqdm


def apply_pool(func,iterable,iterable_size,processes=100):
    with multiprocessing.Pool(processes=processes) as pool:
        result_list = []
        for _, res in tqdm.tqdm(enumerate(pool.imap_unordered(func, iterable)),total=iterable_size):
            result_list.append(res)
    return result_list

# def gen_func(func,state_name):
#     def apply_func(kwargs):    
#         kwargs[state_name] = state_global
#         return func(kwargs)
#     return apply_func


# def apply_pool_with_state(func,iterable,iterable_size,gen_state,state_name):
#     def set_global_object(gen_state):
#         global state_global
        
#         state_global = gen_state()
    
#     apply_func = gen_func(func,state_name)
#     pool = multiprocessing.Pool(processes=os.cpu_count()-40,initializer=set_global_object,initargs=(gen_state,))
#     result_list = []
#     for _, res in tqdm.tqdm(enumerate(pool.imap_unordered(apply_func, iterable)),total=iterable_size):
#         result_list.append(res)
#     return result_list