def warmup_logging():
    # this snippet below is necessary to avoid double logging and extraneous logging caused by transformers
    import logging
    from transformers import (
        WhisperForConditionalGeneration, 
        WhisperProcessor,
        MimiModel,
        AutoFeatureExtractor
    )
    logging.root.handlers = []
    logging.root.manager.loggerDict = {}
    logging.root.level = logging.root.level
