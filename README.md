# rytp
A tool to extract speech from videos and splice them into something else

# Goal
To have a console line tools to:
- manipulate videos:
	- list and store metadata (mostly URL + "donwloaded or not" flag) on all videos+shorts+livestreams from a single youtube channel;
	- download a video from the stored metadata list or directly provided url (may be from a different channel or non-youtube source). Utilise ytdlp for that;
	- maintain a queue to download videos from youtube, as well as pause the download process;
- extract audio tracks into text. For a given video, run a local speech-to-text model (or several). The result must:
	- be stored locally 
	- have a timestamp for every word retrieved from STT-model
	- have a person mapped for every word (have it editable, since I believe that per-person mapping will be stable within a video – but not across years and different auditoriums/channels)
	- have some spectrogram stored nearby, be it by video or by word. Not sure. The goal is to try to retrieve the closest-sounding words together, rather than the first one
- datamine the extract:
	- take a string as an input
	- depending on level of cohesion the user wants, either find single-word segments, or maximize uninterruptedness of a segment, preferring the longer ones over shorter ones.
	- segments found should be close to each other in audio spectrum, to sound more naturally.
- glue the result together. Given the original videos, the timestamps provided by the datamining stage, and ffmpeg – extract the clips from the originals, and glue them together in order. Also provide a list of original URLs used and timestamps cut out in order they appear in the final result.

# Hardware:
- 32 GiB RAM
- 16 GiB RTX 3080 Laptop GPU
- i7-11800H @ 2.30 GHz