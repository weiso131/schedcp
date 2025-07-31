# FFmpeg Video Transcode

**ID:** `video_transcode`

**Category:** media_processing

**Description:** Video transcoding with one large file dominating processing time

## Dependencies

- ffmpeg
- g++

## Small Setup Commands

```bash
mkdir -p clips out
ffmpeg -f lavfi -i testsrc=duration=30:size=320x240:rate=30 -loglevel quiet clips/short.mp4
cp $ORIGINAL_CWD/assets/video_transcode.cpp .
g++ -o small_video_transcode video_transcode.cpp -lavformat -lavcodec -lavutil -lswscale -lpthread -lm -lz
```

## Large Setup Commands

```bash
mkdir -p clips out
ffmpeg -f lavfi -i testsrc=duration=70:size=1920x1080:rate=30 -loglevel quiet clips/long.mp4
cp $ORIGINAL_CWD/assets/video_transcode.cpp .
g++ -o large_video_transcode video_transcode.cpp -lavformat -lavcodec -lavutil -lswscale -lpthread -lm -lz
```

## Small Execution Commands

```bash
./small_video_transcode clips/short.mp4 out/short_out.mp4 640
```

## Large Execution Commands

```bash
./large_video_transcode clips/long.mp4 out/long_out.mp4 640
```

## Cleanup Commands

```bash
rm -rf clips/ out/
rm -f small_video_transcode large_video_transcode video_transcode.cpp
```
