'''
Outline 1:
- Label index as file 1 2 3 as each number corresponding to 5 seconds
    + 1 -> 0:00 - 0:05
    + 2 -> 0:01 - 0:06
    etc...
- Requirements to be a song ( more than 30 seconds )
- Algorithm:
    + Mark song file as T
    + For every consecutive file, mark as [T, T ,T ,T, T]
    + Concatenate all true timestamp together so that 0:00 -> 0:20
    + If between chunk of timestamp, exist a chunk of talking ( around 5second), discard that chunk
      else concatenate the song
    + Result would look something like for a stream of big video [0:00 -> 0:30, 1:00 -> 2:30, 3:40 -> 5:55] etc....
    + Run each fragment with +- 5% of error for opening and ending, in addition, insert fragment to look up for song database.
    + Export the song
    
'''