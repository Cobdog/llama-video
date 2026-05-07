"""Tests for video segmentation."""

from llama_video.segmenter import VideoChunk, segment_video


class TestSegmentVideo:
    """Test video chunking logic."""

    def test_short_video_single_chunk(self):
        chunks = segment_video(total_duration=30.0, chunk_seconds=60.0)
        assert len(chunks) == 1
        assert chunks[0].start_seconds == 0.0
        assert chunks[0].end_seconds == 30.0

    def test_exact_boundary(self):
        chunks = segment_video(total_duration=60.0, chunk_seconds=60.0)
        assert len(chunks) == 1
        assert chunks[0].duration == 60.0

    def test_two_chunks(self):
        chunks = segment_video(total_duration=90.0, chunk_seconds=60.0)
        assert len(chunks) == 2
        assert chunks[0].start_seconds == 0.0
        assert chunks[0].end_seconds == 60.0
        assert chunks[1].start_seconds == 60.0
        assert chunks[1].end_seconds == 90.0

    def test_short_last_chunk(self):
        chunks = segment_video(total_duration=100.0, chunk_seconds=60.0)
        assert len(chunks) == 2
        assert chunks[0].duration == 60.0
        assert chunks[1].duration == 40.0

    def test_many_chunks(self):
        chunks = segment_video(total_duration=180.0, chunk_seconds=60.0)
        assert len(chunks) == 3
        assert all(c.duration == 60.0 for c in chunks)

    def test_max_chunk_cap(self):
        chunks = segment_video(
            total_duration=120.0,
            chunk_seconds=90.0,
            max_chunk_seconds=60.0,
        )
        assert len(chunks) == 2
        assert chunks[0].duration == 60.0

    def test_max_chunk_larger_than_requested(self):
        chunks = segment_video(
            total_duration=90.0,
            chunk_seconds=30.0,
            max_chunk_seconds=60.0,
        )
        assert len(chunks) == 3
        assert chunks[0].duration == 30.0

    def test_chunk_indices_sequential(self):
        chunks = segment_video(total_duration=180.0, chunk_seconds=60.0)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_zero_duration(self):
        chunks = segment_video(total_duration=0.0, chunk_seconds=60.0)
        assert len(chunks) == 1
        assert chunks[0].duration == 0.0


class TestVideoChunk:
    def test_duration_property(self):
        chunk = VideoChunk(start_seconds=10.0, end_seconds=40.0, index=0)
        assert chunk.duration == 30.0
