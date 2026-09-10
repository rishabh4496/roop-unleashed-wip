const FALLBACK_CODECS_BY_FORMAT = {
  avi: ['libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc'],
  mkv: ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'],
  mp4: ['libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc'],
  webm: ['libvpx-vp9'],
};

const CPU_PRESETS = ['auto', 'ultrafast', 'superfast', 'veryfast', 'faster',
  'fast', 'medium', 'slow', 'slower', 'veryslow'];
const NVENC_PRESETS = ['auto', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'];

export const isNvencCodec = (codec) => /_nvenc$/i.test(codec || '');

export const videoQualityMax = (codec) => (
  /libvpx|vp9|aom|av1/i.test(codec || '') ? 63 : 51
);

export function codecsForFormat(meta = {}, format = 'mp4') {
  const installed = Array.isArray(meta.video_codecs) ? meta.video_codecs : [];
  const fromBackend = meta.video_codecs_by_format?.[format];
  const declared = Array.isArray(fromBackend)
    ? fromBackend
    : (FALLBACK_CODECS_BY_FORMAT[format] || installed);
  return installed.length > 0
    ? declared.filter((codec) => installed.includes(codec))
    : declared;
}

export function encoderPresetsForCodec(meta = {}, codec = '') {
  if (isNvencCodec(codec)) {
    return meta.nvenc_encoder_presets || NVENC_PRESETS;
  }
  return meta.encoder_presets || CPU_PRESETS;
}

/** Keep all dependent output controls valid as one atomic state transition. */
export function normalizeOutputSettings(settings, meta = {}) {
  const next = { ...settings };
  const formats = Array.isArray(meta.video_formats) ? meta.video_formats : [];

  if (formats.length > 0 && !formats.includes(next.output_video_format)) {
    next.output_video_format = formats.includes('mp4') ? 'mp4' : formats[0];
  }

  const compatible = codecsForFormat(meta, next.output_video_format || 'mp4');
  if (compatible.length > 0 && !compatible.includes(next.output_video_codec)) {
    next.output_video_codec = compatible[0];
  }

  const quality = Number(next.video_quality);
  const safeQuality = Number.isFinite(quality) ? Math.round(quality) : 14;
  next.video_quality = Math.max(
    0,
    Math.min(safeQuality, videoQualityMax(next.output_video_codec)),
  );

  const presets = encoderPresetsForCodec(meta, next.output_video_codec);
  if (!presets.includes(next.perf_encoder_preset)) next.perf_encoder_preset = 'auto';
  return next;
}

export function outputSettingsChanged(before, after) {
  return ['output_video_format', 'output_video_codec', 'video_quality', 'perf_encoder_preset']
    .some((key) => before?.[key] !== after?.[key]);
}
