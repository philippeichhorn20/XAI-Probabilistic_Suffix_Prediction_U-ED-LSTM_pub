




def get_final_process_step(whole_process_prefix):
    cat_prefixes, num_prefixes = whole_process_prefix
    cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
    num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
    sos_event = [cat_sos_events, num_sos_events]
    return sos_event
