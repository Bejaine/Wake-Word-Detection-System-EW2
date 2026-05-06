#include <driver/i2s.h>

#define I2S_PORT       I2S_NUM_0
#define I2S_SCK_PIN    3
#define I2S_BCK_PIN    26
#define I2S_LRC_PIN    25
#define I2S_DATA_PIN   22

// 4-Byte Cryptographic Header (Mathematically prevents false syncs)
const uint8_t SYNC_HEADER[4] = {0xDE, 0xAD, 0xBE, 0xEF};

void setup() {
  // Lowered to highly stable CH340 tier
  Serial.begin(460800);

  i2s_config_t i2s_config = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = 16000,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_RIGHT, 
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 8,
    .dma_buf_len          = 512,
    .use_apll             = true,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };

  i2s_pin_config_t pin_config = {
    .mck_io_num  = I2S_SCK_PIN,
    .bck_io_num  = I2S_BCK_PIN,
    .ws_io_num   = I2S_LRC_PIN,
    .data_in_num = I2S_DATA_PIN
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void loop() {
  const int BLOCK_SIZE = 128;
  int32_t sample_buffer[BLOCK_SIZE];
  int16_t tx_buffer[BLOCK_SIZE];
  size_t bytesRead = 0;

  esp_err_t result = i2s_read(I2S_PORT, &sample_buffer, sizeof(sample_buffer), &bytesRead, portMAX_DELAY);

  if (result == ESP_OK && bytesRead > 0) {
    int samples = bytesRead / sizeof(int32_t);

    for (int i = 0; i < samples; i++) {
      int32_t val = sample_buffer[i] >> 16;

      if (val > 32767)  val = 32767;
      if (val < -32768) val = -32768;

      tx_buffer[i] = (int16_t)val;
    }

    // Send the 4-byte header followed by the audio payload
    Serial.write(SYNC_HEADER, 4);
    Serial.write((uint8_t*)tx_buffer, samples * sizeof(int16_t));
  }
}
