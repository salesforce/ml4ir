from tensorflow.keras import callbacks


class DebuggingCallback(callbacks.Callback):
    def __init__(self, logger, logging_frequency=25):
        super(DebuggingCallback, self).__init__()

        self.logger = logger
        self.epoch = 0
        self.logging_frequency = logging_frequency

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.logging_frequency == 0:
            self.logger.info("[epoch: {} | batch: {}] {}".format(self.epoch, batch, logs))

    def on_predict_batch_end(self, batch, logs=None):
        if batch % self.logging_frequency == 0:
            self.logger.info("[batch: {}] {}".format(batch, logs))

    def on_test_batch_end(self, batch, logs=None):
        if batch % self.logging_frequency == 0:
            self.logger.info("[batch: {}] {}".format(batch, logs))

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info("End of Epoch {}".format(self.epoch))
        self.logger.info(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.logger.info("Starting Epoch : {}".format(self.epoch))
        self.logger.info(logs)

    def on_train_begin(self, logs):
        self.logger.info("Training Model")

    def on_test_begin(self, logs):
        self.logger.info("Evaluating Model")

    def on_predict_begin(self, logs):
        self.logger.info("Predicting scores using model")

    def on_train_end(self, logs):
        self.logger.info("Completed training model")
        self.logger.info(logs)

    def on_test_end(self, logs):
        self.logger.info("Completed evaluating model")
        self.logger.info(logs)

    def on_predict_end(self, logs):
        self.logger.info("Completed Predicting scores using model")
        self.logger.info(logs)
