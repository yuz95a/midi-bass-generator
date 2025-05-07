def remove_metadata_in_track(track):
    if len(track) >= 2:
        if track[0].is_meta and track[1].type == 'program_change':
            del track[0:2]
    if track:
        if track[-1].is_meta and track[-1].type == 'end_of_track':
            del track[-1]