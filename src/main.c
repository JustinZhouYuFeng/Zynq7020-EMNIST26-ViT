#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xstatus.h"
#include "xemacps.h"
#include "xemacps_hw.h"
#include "xuartps_hw.h"
#include "xtime_l.h"
#define USE_LEGACY_SPLIT_PL 0
#define DEBUG_FUSED_LAYER_COMPARE 0
#define DEBUG_FUSED_IP_STAGE 0
#if USE_LEGACY_SPLIT_PL
#include "xvit_qkv_linear.h"
#endif
#include "netif/xadapter.h"
#include "lwip/init.h"
#include "lwip/inet.h"
#include "lwip/pbuf.h"
#include "lwip/priv/tcp_priv.h"
#include "lwip/udp.h"
#include "platform.h"
#include "emnist_pair_resolvers_vitis.h"
#include "qkv_params_vitis.h"
#include "tinyvit_samples_vitis.h"

#define UART1_BASE XPAR_PS7_UART_1_BASEADDR
#define DEFAULT_IP_ADDRESS "192.168.1.10"
#define DEFAULT_IP_MASK "255.255.255.0"
#define DEFAULT_GW_ADDRESS "192.168.1.1"
#define UDP_IMAGE_PORT 5001
#define UDP_IMAGE_BYTES 784
#ifdef TINYVIT_INPUT_MEAN
#define VIT_INPUT_MEAN TINYVIT_INPUT_MEAN
#else
#define VIT_INPUT_MEAN 0.1307f
#endif

#ifdef TINYVIT_INPUT_STD
#define VIT_INPUT_STD TINYVIT_INPUT_STD
#else
#define VIT_INPUT_STD 0.3081f
#endif
#define TOKENS 17
#define EMBED_DIM 64
#define QKV_DIM 192
#ifdef TINYVIT_NUM_HEADS
#define HEADS TINYVIT_NUM_HEADS
#else
#define HEADS 4
#endif
#define HEAD_DIM (EMBED_DIM / HEADS)
#ifdef TINYVIT_MLP_DIM
#define MLP_DIM TINYVIT_MLP_DIM
#else
#define MLP_DIM 128
#endif
#ifdef TINYVIT_DEPTH
#define VIT_DEPTH TINYVIT_DEPTH
#else
#define VIT_DEPTH 1
#endif
#ifdef TINYVIT_NUM_CLASSES
#define CLASSES TINYVIT_NUM_CLASSES
#else
#define CLASSES 10
#endif
#define QKV_TIMEOUT 10000000U
#define LINEAR_TIMEOUT 20000000U
#define MLP_FUSED_TIMEOUT 200000000U
#define TRANSFORMER_FUSED_TIMEOUT 300000000U
#define UART_TX_TIMEOUT 1000000U

#define LINEAR_BASE 0x44A10000U
#define LINEAR_AP_CTRL 0x00U
#define LINEAR_X_DATA 0x10U
#define LINEAR_W_DATA 0x1CU
#define LINEAR_BIAS_DATA 0x28U
#define LINEAR_Y_DATA 0x34U
#define LINEAR_IN_DIM_DATA 0x40U
#define LINEAR_OUT_DIM_DATA 0x48U

#define GELU_BASE 0x44A20000U
#define GELU_AP_CTRL 0x00U
#define GELU_X_DATA 0x10U
#define GELU_LUT_DATA 0x1CU
#define GELU_Y_DATA 0x28U
#define GELU_LUT_MIN_DATA 0x34U
#define GELU_INDEX_SCALE_DATA 0x3CU
#define GELU_TIMEOUT 20000000U
#define GELU_LUT_ENTRIES 256
#define GELU_LUT_MIN_VALUE (-8.0f)
#define GELU_LUT_MAX_VALUE 8.0f

#define MLP_FUSED_BASE 0x44A30000U
#define MLP_FUSED_AP_CTRL 0x00U
#define MLP_FUSED_X_DATA 0x10U
#define MLP_FUSED_FC1_W_DATA 0x1CU
#define MLP_FUSED_FC1_B_DATA 0x28U
#define MLP_FUSED_GELU_LUT_DATA 0x34U
#define MLP_FUSED_FC2_W_DATA 0x40U
#define MLP_FUSED_FC2_B_DATA 0x4CU
#define MLP_FUSED_Y_DATA 0x58U
#define MLP_FUSED_FC1_SCALE_DATA 0x64U
#define MLP_FUSED_HIDDEN_INV_SCALE_DATA 0x6CU
#define MLP_FUSED_FC2_OUTPUT_SCALE_DATA 0x74U
#define MLP_FUSED_LUT_MIN_DATA 0x7CU
#define MLP_FUSED_LUT_INDEX_SCALE_DATA 0x84U
#define MLP_FUSED_HIDDEN_MAX_CALIB 2.4617166519f
#define MLP_FUSED_HIDDEN_INV_SCALE (127.0f / MLP_FUSED_HIDDEN_MAX_CALIB)
#define MAYBE_UNUSED __attribute__((unused))

#ifdef XPAR_VIT_TLAYER_FUSED_0_S_AXI_CONTROL_BASEADDR
#define TRANSFORMER_FUSED_BASE XPAR_VIT_TLAYER_FUSED_0_S_AXI_CONTROL_BASEADDR
#else
#define TRANSFORMER_FUSED_BASE 0x44A00000U
#endif
#define TRANSFORMER_FUSED_AP_CTRL 0x00U
#define TRANSFORMER_FUSED_TOKEN_IN_DATA 0x10U
#define TRANSFORMER_FUSED_NORM1_W_DATA 0x1CU
#define TRANSFORMER_FUSED_NORM1_B_DATA 0x28U
#define TRANSFORMER_FUSED_QKV_W_DATA 0x34U
#define TRANSFORMER_FUSED_QKV_B_DATA 0x40U
#define TRANSFORMER_FUSED_ATTN_PROJ_W_DATA 0x4CU
#define TRANSFORMER_FUSED_ATTN_PROJ_B_DATA 0x58U
#define TRANSFORMER_FUSED_NORM2_W_DATA 0x64U
#define TRANSFORMER_FUSED_NORM2_B_DATA 0x70U
#define TRANSFORMER_FUSED_FC1_W_DATA 0x7CU
#define TRANSFORMER_FUSED_FC1_B_DATA 0x88U
#define TRANSFORMER_FUSED_GELU_LUT_DATA 0x94U
#define TRANSFORMER_FUSED_FC2_W_DATA 0xA0U
#define TRANSFORMER_FUSED_FC2_B_DATA 0xACU
#define TRANSFORMER_FUSED_TOKEN_OUT_DATA 0xB8U
#define TRANSFORMER_FUSED_QKV_W_SCALE_DATA 0xC4U
#define TRANSFORMER_FUSED_ATTN_PROJ_W_SCALE_DATA 0xCCU
#define TRANSFORMER_FUSED_FC1_W_SCALE_DATA 0xD4U
#define TRANSFORMER_FUSED_FC2_W_SCALE_DATA 0xDCU
#define TRANSFORMER_FUSED_HIDDEN_INV_SCALE_DATA 0xE4U
#define TRANSFORMER_FUSED_LUT_MIN_DATA 0xECU
#define TRANSFORMER_FUSED_LUT_INDEX_SCALE_DATA 0xF4U

#define X_BUF_ADDR      0x01000000U
#define W_BUF_ADDR      0x01100000U
#define BIAS_BUF_ADDR   0x01200000U
#define Y_BUF_ADDR      0x01300000U

#define LIN_X_BUF_ADDR      0x01400000U
#define LIN_W_BUF_ADDR      0x01500000U
#define LIN_BIAS_BUF_ADDR   0x01600000U
#define LIN_Y_BUF_ADDR      0x01700000U

#define GELU_X_BUF_ADDR     0x01800000U
#define GELU_LUT_BUF_ADDR   0x01900000U
#define GELU_Y_BUF_ADDR     0x01A00000U

#define MLP_FUSED_X_BUF_ADDR       0x02000000U
#define MLP_FUSED_FC1_W_BUF_ADDR   0x02100000U
#define MLP_FUSED_FC1_B_BUF_ADDR   0x02200000U
#define MLP_FUSED_LUT_BUF_ADDR     0x02300000U
#define MLP_FUSED_FC2_W_BUF_ADDR   0x02400000U
#define MLP_FUSED_FC2_B_BUF_ADDR   0x02500000U
#define MLP_FUSED_Y_BUF_ADDR       0x02600000U
#define FUSED_ATTN_PROJ_W_BUF_ADDR 0x02700000U

#define FUSED_TOKEN_A_BUF_ADDR     0x03000000U
#define FUSED_TOKEN_B_BUF_ADDR     0x03100000U
#define FUSED_PARAM_BUF_ADDR       0x03200000U

#define X_BUF_SIZE      (TOKENS * EMBED_DIM * sizeof(signed char))
#define W_BUF_SIZE      (EMBED_DIM * QKV_DIM * sizeof(signed char))
#define BIAS_BUF_SIZE   (QKV_DIM * sizeof(int))
#define Y_BUF_SIZE      (TOKENS * QKV_DIM * sizeof(int))
#define QKV_W_LAYER_ADDR(layer) (W_BUF_ADDR + ((u32)(layer) * W_BUF_SIZE))
#define FUSED_QKV_W_BUF_SIZE (EMBED_DIM * QKV_DIM * sizeof(float))
#define FUSED_QKV_W_LAYER_ADDR(layer) (W_BUF_ADDR + ((u32)(layer) * FUSED_QKV_W_BUF_SIZE))

#define LIN_IN_MAX      128
#define LIN_OUT_MAX     192
#define LIN_X_BUF_SIZE      (TOKENS * LIN_IN_MAX * sizeof(signed char))
#define LIN_W_BUF_SIZE      (LIN_IN_MAX * LIN_OUT_MAX * sizeof(signed char))
#define LIN_BIAS_BUF_SIZE   (LIN_OUT_MAX * sizeof(int))
#define LIN_Y_BUF_SIZE      (TOKENS * LIN_OUT_MAX * sizeof(int))
#define LIN_STATIC_ATTN_OUT_W_ADDR  (LIN_W_BUF_ADDR + 0x00020000U)
#define LIN_STATIC_HEAD_W_ADDR      (LIN_W_BUF_ADDR + 0x00030000U)
#define LIN_ATTN_OUT_W_LAYER_ADDR(layer) \
    (LIN_STATIC_ATTN_OUT_W_ADDR + ((u32)(layer) * EMBED_DIM * EMBED_DIM * sizeof(signed char)))
#define GELU_X_BUF_SIZE     (TOKENS * MLP_DIM * sizeof(float))
#define GELU_LUT_BUF_SIZE   (GELU_LUT_ENTRIES * sizeof(float))
#define GELU_Y_BUF_SIZE     (TOKENS * MLP_DIM * sizeof(float))
#define MLP_FUSED_X_BUF_SIZE       (TOKENS * EMBED_DIM * sizeof(signed char))
#define MLP_FUSED_FC1_W_BUF_SIZE   (EMBED_DIM * MLP_DIM * sizeof(signed char))
#define MLP_FUSED_FC1_B_BUF_SIZE   (MLP_DIM * sizeof(int))
#define MLP_FUSED_LUT_BUF_SIZE     (GELU_LUT_ENTRIES * sizeof(float))
#define MLP_FUSED_FC2_W_BUF_SIZE   (MLP_DIM * EMBED_DIM * sizeof(signed char))
#define MLP_FUSED_FC2_B_BUF_SIZE   (EMBED_DIM * sizeof(int))
#define MLP_FUSED_Y_BUF_SIZE       (TOKENS * EMBED_DIM * sizeof(float))
#define FUSED_ATTN_PROJ_W_BUF_SIZE (EMBED_DIM * EMBED_DIM * sizeof(signed char))
#define FUSED_ATTN_PROJ_W_FLOAT_BUF_SIZE (EMBED_DIM * EMBED_DIM * sizeof(float))
#define MLP_FUSED_FC1_W_FLOAT_BUF_SIZE   (EMBED_DIM * MLP_DIM * sizeof(float))
#define MLP_FUSED_FC2_W_FLOAT_BUF_SIZE   (MLP_DIM * EMBED_DIM * sizeof(float))
#define MLP_FUSED_FC1_W_LAYER_ADDR(layer) \
    (MLP_FUSED_FC1_W_BUF_ADDR + ((u32)(layer) * MLP_FUSED_FC1_W_FLOAT_BUF_SIZE))
#define MLP_FUSED_FC2_W_LAYER_ADDR(layer) \
    (MLP_FUSED_FC2_W_BUF_ADDR + ((u32)(layer) * MLP_FUSED_FC2_W_FLOAT_BUF_SIZE))
#define FUSED_ATTN_PROJ_W_LAYER_ADDR(layer) \
    (FUSED_ATTN_PROJ_W_BUF_ADDR + ((u32)(layer) * FUSED_ATTN_PROJ_W_FLOAT_BUF_SIZE))
#define FUSED_TOKEN_BUF_SIZE       (TOKENS * EMBED_DIM * sizeof(float))
#define FUSED_NORM_BLOCK_SIZE      (VIT_DEPTH * EMBED_DIM * sizeof(float))
#define FUSED_QKV_B_BLOCK_SIZE     (VIT_DEPTH * QKV_DIM * sizeof(float))
#define FUSED_FC1_B_BLOCK_SIZE     (VIT_DEPTH * MLP_DIM * sizeof(float))
#define FUSED_NORM1_W_ADDR         (FUSED_PARAM_BUF_ADDR)
#define FUSED_NORM1_B_ADDR         (FUSED_NORM1_W_ADDR + FUSED_NORM_BLOCK_SIZE)
#define FUSED_QKV_B_ADDR           (FUSED_NORM1_B_ADDR + FUSED_NORM_BLOCK_SIZE)
#define FUSED_ATTN_PROJ_B_ADDR     (FUSED_QKV_B_ADDR + FUSED_QKV_B_BLOCK_SIZE)
#define FUSED_NORM2_W_ADDR         (FUSED_ATTN_PROJ_B_ADDR + FUSED_NORM_BLOCK_SIZE)
#define FUSED_NORM2_B_ADDR         (FUSED_NORM2_W_ADDR + FUSED_NORM_BLOCK_SIZE)
#define FUSED_FC1_B_ADDR           (FUSED_NORM2_B_ADDR + FUSED_NORM_BLOCK_SIZE)
#define FUSED_FC2_B_ADDR           (FUSED_FC1_B_ADDR + FUSED_FC1_B_BLOCK_SIZE)
#define FUSED_LAYER_NORM_ADDR(base, layer) ((base) + ((u32)(layer) * EMBED_DIM * sizeof(float)))
#define FUSED_LAYER_QKV_B_ADDR(layer) (FUSED_QKV_B_ADDR + ((u32)(layer) * QKV_DIM * sizeof(float)))
#define FUSED_LAYER_FC1_B_ADDR(layer) (FUSED_FC1_B_ADDR + ((u32)(layer) * MLP_DIM * sizeof(float)))

static signed char (*const x_buf)[EMBED_DIM] =
    (signed char (*)[EMBED_DIM])X_BUF_ADDR;
static signed char (*const w_buf)[QKV_DIM] =
    (signed char (*)[QKV_DIM])W_BUF_ADDR;
static int *const bias_buf = (int *)BIAS_BUF_ADDR;
static int (*const y_buf)[QKV_DIM] = (int (*)[QKV_DIM])Y_BUF_ADDR;
static signed char (*const lin_x_buf)[LIN_IN_MAX] =
    (signed char (*)[LIN_IN_MAX])LIN_X_BUF_ADDR;
static signed char (*const lin_w_buf)[LIN_OUT_MAX] =
    (signed char (*)[LIN_OUT_MAX])LIN_W_BUF_ADDR;
static int *const lin_bias_buf = (int *)LIN_BIAS_BUF_ADDR;
static int (*const lin_y_buf)[LIN_OUT_MAX] = (int (*)[LIN_OUT_MAX])LIN_Y_BUF_ADDR;
static float (*const gelu_x_buf)[MLP_DIM] = (float (*)[MLP_DIM])GELU_X_BUF_ADDR;
static float *const gelu_lut_buf = (float *)GELU_LUT_BUF_ADDR;
static float (*const gelu_y_buf)[MLP_DIM] = (float (*)[MLP_DIM])GELU_Y_BUF_ADDR;
static signed char (*const mlp_fused_x_buf)[EMBED_DIM] =
    (signed char (*)[EMBED_DIM])MLP_FUSED_X_BUF_ADDR;
static signed char (*const mlp_fused_fc1_w_buf)[MLP_DIM] =
    (signed char (*)[MLP_DIM])MLP_FUSED_FC1_W_BUF_ADDR;
static int *const mlp_fused_fc1_b_buf = (int *)MLP_FUSED_FC1_B_BUF_ADDR;
static float *const mlp_fused_lut_buf = (float *)MLP_FUSED_LUT_BUF_ADDR;
static signed char (*const mlp_fused_fc2_w_buf)[EMBED_DIM] =
    (signed char (*)[EMBED_DIM])MLP_FUSED_FC2_W_BUF_ADDR;
static int *const mlp_fused_fc2_b_buf = (int *)MLP_FUSED_FC2_B_BUF_ADDR;
static float (*const mlp_fused_y_buf)[EMBED_DIM] =
    (float (*)[EMBED_DIM])MLP_FUSED_Y_BUF_ADDR;
static signed char (*const fused_attn_proj_w_buf)[EMBED_DIM] =
    (signed char (*)[EMBED_DIM])FUSED_ATTN_PROJ_W_BUF_ADDR;
static float (*const fused_token_a_buf)[EMBED_DIM] =
    (float (*)[EMBED_DIM])FUSED_TOKEN_A_BUF_ADDR;
static float (*const fused_token_b_buf)[EMBED_DIM] =
    (float (*)[EMBED_DIM])FUSED_TOKEN_B_BUF_ADDR;
static float *const fused_norm1_w_buf = (float *)FUSED_NORM1_W_ADDR;
static float *const fused_norm1_b_buf = (float *)FUSED_NORM1_B_ADDR;
static float *const fused_qkv_b_buf = (float *)FUSED_QKV_B_ADDR;
static float *const fused_attn_proj_b_buf = (float *)FUSED_ATTN_PROJ_B_ADDR;
static float *const fused_norm2_w_buf = (float *)FUSED_NORM2_W_ADDR;
static float *const fused_norm2_b_buf = (float *)FUSED_NORM2_B_ADDR;
static float *const fused_fc1_b_buf = (float *)FUSED_FC1_B_ADDR;
static float *const fused_fc2_b_buf = (float *)FUSED_FC2_B_ADDR;

static float token[TOKENS][EMBED_DIM];
static float normed[TOKENS][EMBED_DIM];
static float qkv[TOKENS][QKV_DIM];
static float attn_out[TOKENS][EMBED_DIM];
static float attn_prob[TOKENS][TOKENS];
static float mlp_hidden[TOKENS][MLP_DIM];
static float logits[CLASSES];
static float fused_debug_ref_token[TOKENS][EMBED_DIM];
static float fused_debug_pl_token[TOKENS][EMBED_DIM];
static int gelu_lut_ready = 0;
static u32 last_infer_ms = 0U;
static u32 last_patch_ms = 0U;
static u32 last_stack_ms = 0U;
static u32 last_head_ms = 0U;
static u32 prof_norm_ms = 0U;
static u32 prof_qkv_ms = 0U;
static u32 prof_attn_ms = 0U;
static u32 prof_proj_ms = 0U;
static u32 prof_mlp_ms = 0U;
struct netif server_netif;
static volatile int udp_image_ready = 0;
static volatile u16_t udp_image_len = 0;
static unsigned char udp_image_raw[UDP_IMAGE_BYTES];
static float udp_image_float[UDP_IMAGE_BYTES];
volatile u32 boot_marker = 0U;
volatile u32 boot_counter = 0U;
static XEmacPs mdio_diag_emac;
static int mdio_diag_ready = 0;
static u32 mdio_diag_phy_addr = 0xFFFFFFFFU;
static u16 mdio_last_status = 0xFFFFU;
static u32 last_rx_frames = 0xFFFFFFFFU;
static u32 last_tx_frames = 0xFFFFFFFFU;

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;

static void uart1_init_direct(void)
{
    Xil_Out32(UART1_BASE + XUARTPS_CR_OFFSET, 0x00000003);
    Xil_Out32(UART1_BASE + XUARTPS_MR_OFFSET, 0x00000020);
    Xil_Out32(UART1_BASE + XUARTPS_BAUDGEN_OFFSET, 0x0000007C);
    Xil_Out32(UART1_BASE + XUARTPS_BAUDDIV_OFFSET, 0x00000006);
    Xil_Out32(UART1_BASE + XUARTPS_CR_OFFSET, 0x00000014);
}

static void uart1_putc(char c)
{
    u32 timeout = UART_TX_TIMEOUT;

    while (XUartPs_IsTransmitFull(UART1_BASE) && timeout > 0U) {
        timeout--;
    }

    if (timeout != 0U) {
        XUartPs_WriteReg(UART1_BASE, XUARTPS_FIFO_OFFSET, (u32)c);
    }
}

static void uart1_puts(const char *s)
{
    while (*s != '\0') {
        uart1_putc(*s++);
    }
}

static void uart1_put_hex(u32 v)
{
    static const char digits[] = "0123456789ABCDEF";

    uart1_puts("0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart1_putc(digits[(v >> shift) & 0xFU]);
    }
}

static u32 float_bits(float v)
{
    union {
        float f;
        u32 u;
    } conv;

    conv.f = v;
    return conv.u;
}

static void stage(const char *s)
{
    uart1_puts(s);
    uart1_puts("\r\n");
    for (volatile int delay = 0; delay < 300000; delay++);
}

static void scan_mdio_phy(void)
{
    XEmacPs_Config *cfg;
    u32 base;
    u32 nwctrl;
    u16 id1;
    u16 id2;
    u16 status;
    u16 control;
    int found = 0;

    uart1_puts("MDIO_SCAN_BEGIN\r\n");
    cfg = XEmacPs_LookupConfig(XPAR_PS7_ETHERNET_0_DEVICE_ID);
    if (cfg == 0) {
        uart1_puts("MDIO_CFG_MISSING\r\n");
        return;
    }

    if (XEmacPs_CfgInitialize(&mdio_diag_emac, cfg, cfg->BaseAddress) != XST_SUCCESS) {
        uart1_puts("MDIO_CFG_INIT_FAIL\r\n");
        return;
    }
    mdio_diag_ready = 1;

    XEmacPs_SetMdioDivisor(&mdio_diag_emac, MDC_DIV_224);
    base = cfg->BaseAddress;
    uart1_puts("GEM_BASE=");
    uart1_put_hex(base);
    uart1_puts(" NWCTRL_BEFORE=");
    uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_NWCTRL_OFFSET));
    uart1_puts(" NWCFG=");
    uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_NWCFG_OFFSET));
    uart1_puts(" NWSR=");
    uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_NWSR_OFFSET));
    uart1_puts("\r\n");

    nwctrl = XEmacPs_ReadReg(base, XEMACPS_NWCTRL_OFFSET);
    XEmacPs_WriteReg(base, XEMACPS_NWCTRL_OFFSET,
                     nwctrl | XEMACPS_NWCTRL_MDEN_MASK);
    uart1_puts("NWCTRL_AFTER_MDIO_EN=");
    uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_NWCTRL_OFFSET));
    uart1_puts("\r\n");

    for (u32 addr = 0U; addr < 32U; addr++) {
        id1 = 0xFFFFU;
        id2 = 0xFFFFU;
        status = 0xFFFFU;
        control = 0xFFFFU;
        (void)XEmacPs_PhyRead(&mdio_diag_emac, addr, 0U, &control);
        (void)XEmacPs_PhyRead(&mdio_diag_emac, addr, 1U, &status);
        (void)XEmacPs_PhyRead(&mdio_diag_emac, addr, 2U, &id1);
        (void)XEmacPs_PhyRead(&mdio_diag_emac, addr, 3U, &id2);

        if (!((id1 == 0xFFFFU && id2 == 0xFFFFU) || (id1 == 0U && id2 == 0U))) {
            found = 1;
            if (mdio_diag_phy_addr == 0xFFFFFFFFU || (mdio_diag_phy_addr == 0U && addr != 0U)) {
                mdio_diag_phy_addr = addr;
            }
            uart1_puts("PHY addr=");
            uart1_put_hex(addr);
            uart1_puts(" ctrl=");
            uart1_put_hex(control);
            uart1_puts(" stat=");
            uart1_put_hex(status);
            uart1_puts(" id1=");
            uart1_put_hex(id1);
            uart1_puts(" id2=");
            uart1_put_hex(id2);
            uart1_puts("\r\n");
        }
    }
    if (!found) {
        uart1_puts("PHY_NOT_FOUND_BY_MDIO\r\n");
    }
    uart1_puts("MDIO_SCAN_END\r\n");
}

static void poll_mdio_link_status(void)
{
    u16 status1 = 0xFFFFU;
    u16 status2 = 0xFFFFU;

    if (!mdio_diag_ready || mdio_diag_phy_addr == 0xFFFFFFFFU) {
        return;
    }

    (void)XEmacPs_PhyRead(&mdio_diag_emac, mdio_diag_phy_addr, 1U, &status1);
    (void)XEmacPs_PhyRead(&mdio_diag_emac, mdio_diag_phy_addr, 1U, &status2);

    if (status2 != mdio_last_status || ((boot_counter & 0x3FFFFFU) == 0U)) {
        mdio_last_status = status2;
        uart1_puts("PHY_LINK addr=");
        uart1_put_hex(mdio_diag_phy_addr);
        uart1_puts(" stat=");
        uart1_put_hex(status2);
        uart1_puts(" link=");
        uart1_put_hex((status2 & 0x0004U) ? 1U : 0U);
        uart1_puts(" an=");
        uart1_put_hex((status2 & 0x0020U) ? 1U : 0U);
        uart1_puts("\r\n");
    }
}

static void poll_gem_stats(void)
{
    u32 base = XPAR_PS7_ETHERNET_0_BASEADDR;
    u32 rx_frames = XEmacPs_ReadReg(base, XEMACPS_RXCNT_OFFSET);
    u32 tx_frames = XEmacPs_ReadReg(base, XEMACPS_TXCNT_OFFSET);

    if (rx_frames != last_rx_frames || tx_frames != last_tx_frames ||
        ((boot_counter & 0x7FFFFFU) == 0U)) {
        last_rx_frames = rx_frames;
        last_tx_frames = tx_frames;
        uart1_puts("GEM_STATS rx=");
        uart1_put_hex(rx_frames);
        uart1_puts(" tx=");
        uart1_put_hex(tx_frames);
        uart1_puts(" rx_fcs=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_RXFCSCNT_OFFSET));
        uart1_puts(" rx_sym=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_RXSYMBCNT_OFFSET));
        uart1_puts(" rx_align=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_RXALIGNCNT_OFFSET));
        uart1_puts(" isr=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_ISR_OFFSET));
        uart1_puts(" rxsr=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_RXSR_OFFSET));
        uart1_puts(" txsr=");
        uart1_put_hex(XEmacPs_ReadReg(base, XEMACPS_TXSR_OFFSET));
        uart1_puts("\r\n");
    }
}

static void uart1_put_dec_u32(u32 v)
{
    char buf[11];
    int pos = 0;

    if (v == 0U) {
        uart1_putc('0');
        return;
    }

    while (v != 0U && pos < (int)sizeof(buf)) {
        buf[pos++] = (char)('0' + (v % 10U));
        v /= 10U;
    }

    while (pos > 0) {
        uart1_putc(buf[--pos]);
    }
}

static void print_ip(const char *msg, ip_addr_t *ip)
{
    uart1_puts(msg);
    uart1_put_dec_u32((u32)ip4_addr1(ip));
    uart1_putc('.');
    uart1_put_dec_u32((u32)ip4_addr2(ip));
    uart1_putc('.');
    uart1_put_dec_u32((u32)ip4_addr3(ip));
    uart1_putc('.');
    uart1_put_dec_u32((u32)ip4_addr4(ip));
    uart1_puts("\r\n");
}

static void assign_default_ip(ip_addr_t *ip, ip_addr_t *mask, ip_addr_t *gw)
{
    inet_aton(DEFAULT_IP_ADDRESS, ip);
    inet_aton(DEFAULT_IP_MASK, mask);
    inet_aton(DEFAULT_GW_ADDRESS, gw);
}

static void udp_image_recv(void *arg, struct udp_pcb *pcb, struct pbuf *p,
    const ip_addr_t *addr, u16_t port)
{
    const char ack[] = "ACK";
    struct pbuf *ack_buf;
    (void)arg;

    if (p == NULL) {
        return;
    }

    udp_image_len = p->tot_len;
    if (p->tot_len == UDP_IMAGE_BYTES) {
        pbuf_copy_partial(p, udp_image_raw, UDP_IMAGE_BYTES, 0);
        udp_image_ready = 1;
        uart1_puts("UDP_RX_IMAGE len=");
        uart1_put_dec_u32((u32)p->tot_len);
        uart1_puts(" from=");
        uart1_puts(inet_ntoa(*addr));
        uart1_putc(':');
        uart1_put_dec_u32((u32)port);
        uart1_puts("\r\n");
    } else {
        uart1_puts("UDP_RX_BAD_LEN expected=");
        uart1_put_dec_u32(UDP_IMAGE_BYTES);
        uart1_puts(" got=");
        uart1_put_dec_u32((u32)p->tot_len);
        uart1_puts("\r\n");
    }

    ack_buf = pbuf_alloc(PBUF_TRANSPORT, sizeof(ack) - 1, PBUF_RAM);
    if (ack_buf != NULL) {
        pbuf_take(ack_buf, ack, sizeof(ack) - 1);
        udp_sendto(pcb, ack_buf, addr, port);
        pbuf_free(ack_buf);
    }

    pbuf_free(p);
}

static int start_udp_image_server(void)
{
    struct udp_pcb *pcb;
    err_t err;

    pcb = udp_new();
    if (pcb == NULL) {
        uart1_puts("UDP_SERVER_CREATE_FAIL\r\n");
        return -1;
    }

    err = udp_bind(pcb, IP_ADDR_ANY, UDP_IMAGE_PORT);
    if (err != ERR_OK) {
        uart1_puts("UDP_SERVER_BIND_FAIL err=");
        uart1_put_hex((u32)err);
        uart1_puts("\r\n");
        udp_remove(pcb);
        return -2;
    }

    udp_recv(pcb, udp_image_recv, NULL);
    uart1_puts("UDP_SERVER_READY port=");
    uart1_put_dec_u32(UDP_IMAGE_PORT);
    uart1_puts("\r\n");
    return 0;
}

static int net_init(void)
{
    ip_addr_t ipaddr;
    ip_addr_t netmask;
    ip_addr_t gw;
    unsigned char mac_ethernet_address[] = {
        0x00, 0x0a, 0x35, 0x00, 0x01, 0x02
    };

    init_platform();
    lwip_init();
    assign_default_ip(&ipaddr, &netmask, &gw);

    if (!xemac_add(&server_netif, &ipaddr, &netmask, &gw,
            mac_ethernet_address, PLATFORM_EMAC_BASEADDR)) {
        uart1_puts("NETIF_ADD_FAIL\r\n");
        return -1;
    }

    netif_set_default(&server_netif);
    platform_enable_interrupts();
    netif_set_up(&server_netif);

    print_ip("Board IP: ", &server_netif.ip_addr);
    print_ip("Netmask : ", &server_netif.netmask);
    print_ip("Gateway : ", &server_netif.gw);

    return start_udp_image_server();
}

static void net_poll(void)
{
    if (TcpFastTmrFlag) {
        tcp_fasttmr();
        TcpFastTmrFlag = 0;
    }
    if (TcpSlowTmrFlag) {
        tcp_slowtmr();
        TcpSlowTmrFlag = 0;
    }

    xemacif_input(&server_netif);
}

static float fast_sqrtf(float x)
{
    float g = (x > 1.0f) ? x : 1.0f;

    for (int i = 0; i < 10; i++) {
        g = 0.5f * (g + x / g);
    }
    return g;
}

static float fast_expf(float x)
{
    if (x < -10.0f) {
        return 0.0f;
    }
    if (x > 10.0f) {
        x = 10.0f;
    }

    float y = 1.0f + x / 256.0f;
    for (int i = 0; i < 8; i++) {
        y *= y;
    }
    return y;
}

static float fast_tanhf(float x)
{
    float e = fast_expf(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

static float gelu(float x)
{
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + fast_tanhf(inner));
}

static void layer_norm_64(
    const float in[EMBED_DIM],
    const float gamma[EMBED_DIM],
    const float beta[EMBED_DIM],
    float out[EMBED_DIM])
{
    float mean = 0.0f;
    float var = 0.0f;

    for (int i = 0; i < EMBED_DIM; i++) {
        mean += in[i];
    }
    mean /= (float)EMBED_DIM;

    for (int i = 0; i < EMBED_DIM; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)EMBED_DIM;

    float inv = 1.0f / fast_sqrtf(var + 1.0e-5f);
    for (int i = 0; i < EMBED_DIM; i++) {
        out[i] = (in[i] - mean) * inv * gamma[i] + beta[i];
    }
}

static signed char quantize_i8(float x, float scale)
{
    int v;

    if (x >= 0.0f) {
        v = (int)(x / scale + 0.5f);
    } else {
        v = (int)(x / scale - 0.5f);
    }

    if (v > 127) {
        v = 127;
    }
    if (v < -128) {
        v = -128;
    }
    return (signed char)v;
}

static int round_i32(float x)
{
    if (x >= 0.0f) {
        return (int)(x + 0.5f);
    }
    return (int)(x - 0.5f);
}

static float max_abs_attn_out(void)
{
    float max_abs = 0.0f;

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            float v = attn_out[t][e];
            if (v < 0.0f) {
                v = -v;
            }
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }

    return max_abs;
}

static float max_abs_attn_out_weight(void)
{
    float max_abs = 0.0f;

    for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
        float v = TINYVIT_ATTN_OUT_W[i];
        if (v < 0.0f) {
            v = -v;
        }
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

static float max_abs_head_input(void)
{
    float max_abs = 0.0f;

    for (int e = 0; e < EMBED_DIM; e++) {
        float v = normed[0][e];
        if (v < 0.0f) {
            v = -v;
        }
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

static float max_abs_mlp_hidden(void)
{
    float max_abs = 0.0f;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < MLP_DIM; i++) {
            float v = mlp_hidden[t][i];
            if (v < 0.0f) {
                v = -v;
            }
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }

    return max_abs;
}

static float max_abs_fc1_input(void)
{
    float max_abs = 0.0f;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            float v = normed[t][i];
            if (v < 0.0f) {
                v = -v;
            }
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }

    return max_abs;
}

static float max_abs_fc1_weight(void)
{
    float max_abs = 0.0f;

    for (int i = 0; i < MLP_DIM * EMBED_DIM; i++) {
        float v = TINYVIT_FC1_W[i];
        if (v < 0.0f) {
            v = -v;
        }
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

static float max_abs_fc2_weight(void)
{
    float max_abs = 0.0f;

    for (int i = 0; i < EMBED_DIM * MLP_DIM; i++) {
        float v = TINYVIT_FC2_W[i];
        if (v < 0.0f) {
            v = -v;
        }
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

static float max_abs_head_weight(void)
{
    float max_abs = 0.0f;

    for (int i = 0; i < CLASSES * EMBED_DIM; i++) {
        float v = TINYVIT_HEAD_W[i];
        if (v < 0.0f) {
            v = -v;
        }
        if (v > max_abs) {
            max_abs = v;
        }
    }

    return max_abs;
}

static void linear_write64(u32 offset, u32 addr)
{
    Xil_Out32(LINEAR_BASE + offset, addr);
    Xil_Out32(LINEAR_BASE + offset + 4U, 0U);
}

static void gelu_write64(u32 offset, u32 addr)
{
    Xil_Out32(GELU_BASE + offset, addr);
    Xil_Out32(GELU_BASE + offset + 4U, 0U);
}

static void mlp_fused_write64(u32 offset, u32 addr)
{
    Xil_Out32(MLP_FUSED_BASE + offset, addr);
    Xil_Out32(MLP_FUSED_BASE + offset + 4U, 0U);
}

static void transformer_fused_write64(u32 offset, u32 addr)
{
    Xil_Out32(TRANSFORMER_FUSED_BASE + offset, addr);
    Xil_Out32(TRANSFORMER_FUSED_BASE + offset + 4U, 0U);
}

static void prepare_gelu_lut(void)
{
    if (gelu_lut_ready != 0) {
        return;
    }

    for (int i = 0; i < GELU_LUT_ENTRIES; i++) {
        float x = GELU_LUT_MIN_VALUE +
            (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE) *
            (float)i / (float)(GELU_LUT_ENTRIES - 1);
        gelu_lut_buf[i] = gelu(x);
        mlp_fused_lut_buf[i] = gelu_lut_buf[i];
    }

    Xil_DCacheFlushRange((INTPTR)gelu_lut_buf, GELU_LUT_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_lut_buf, MLP_FUSED_LUT_BUF_SIZE);
    gelu_lut_ready = 1;
}

static void preload_pl_static_params(void)
{
    const int norm_total = VIT_DEPTH * EMBED_DIM;
    const int qkv_total = VIT_DEPTH * QKV_DIM;
    const int fc1_total = VIT_DEPTH * MLP_DIM;

    for (int layer = 0; layer < VIT_DEPTH; layer++) {
        float (*fused_qkv_w_layer)[QKV_DIM] =
            (float (*)[QKV_DIM])FUSED_QKV_W_LAYER_ADDR(layer);
        const int qkv_base = layer * QKV_DIM * EMBED_DIM;

        for (int e = 0; e < EMBED_DIM; e++) {
            for (int q = 0; q < QKV_DIM; q++) {
                int src = qkv_base + q * EMBED_DIM + e;
                fused_qkv_w_layer[e][q] = TINYVIT_QKV_W_ALL[src];
            }
        }
        Xil_DCacheFlushRange((INTPTR)fused_qkv_w_layer, FUSED_QKV_W_BUF_SIZE);

        signed char (*attn_w_layer)[LIN_OUT_MAX] =
            (signed char (*)[LIN_OUT_MAX])LIN_ATTN_OUT_W_LAYER_ADDR(layer);
        float (*fused_attn_w_float_layer)[EMBED_DIM] =
            (float (*)[EMBED_DIM])FUSED_ATTN_PROJ_W_LAYER_ADDR(layer);
        const int attn_base = layer * EMBED_DIM * EMBED_DIM;
        for (int i = 0; i < EMBED_DIM; i++) {
            for (int o = 0; o < EMBED_DIM; o++) {
                int src = attn_base + o * EMBED_DIM + i;
                signed char w = TINYVIT_ATTN_OUT_W_I8_ALL[src];
                attn_w_layer[i][o] = w;
                fused_attn_w_float_layer[o][i] = TINYVIT_ATTN_OUT_W_ALL[src];
            }
        }
        Xil_DCacheFlushRange((INTPTR)attn_w_layer, EMBED_DIM * EMBED_DIM * sizeof(signed char));
        Xil_DCacheFlushRange((INTPTR)fused_attn_w_float_layer, FUSED_ATTN_PROJ_W_FLOAT_BUF_SIZE);

        float (*fc1_w_float_layer)[EMBED_DIM] =
            (float (*)[EMBED_DIM])MLP_FUSED_FC1_W_LAYER_ADDR(layer);
        const int fc1_base = layer * MLP_DIM * EMBED_DIM;
        for (int i = 0; i < EMBED_DIM; i++) {
            for (int h = 0; h < MLP_DIM; h++) {
                int src = fc1_base + h * EMBED_DIM + i;
                fc1_w_float_layer[h][i] = TINYVIT_FC1_W_ALL[src];
            }
        }
        Xil_DCacheFlushRange((INTPTR)fc1_w_float_layer, MLP_FUSED_FC1_W_FLOAT_BUF_SIZE);

        float (*fc2_w_float_layer)[MLP_DIM] =
            (float (*)[MLP_DIM])MLP_FUSED_FC2_W_LAYER_ADDR(layer);
        const int fc2_base = layer * EMBED_DIM * MLP_DIM;
        for (int h = 0; h < MLP_DIM; h++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                int src = fc2_base + e * MLP_DIM + h;
                fc2_w_float_layer[e][h] = TINYVIT_FC2_W_ALL[src];
            }
        }
        Xil_DCacheFlushRange((INTPTR)fc2_w_float_layer, MLP_FUSED_FC2_W_FLOAT_BUF_SIZE);
    }

    for (int i = 0; i < norm_total; i++) {
        fused_norm1_w_buf[i] = TINYVIT_NORM1_W_ALL[i];
        fused_norm1_b_buf[i] = TINYVIT_NORM1_B_ALL[i];
        fused_norm2_w_buf[i] = TINYVIT_NORM2_W_ALL[i];
        fused_norm2_b_buf[i] = TINYVIT_NORM2_B_ALL[i];
        fused_attn_proj_b_buf[i] = TINYVIT_ATTN_OUT_B_ALL[i];
        fused_fc2_b_buf[i] = TINYVIT_FC2_B_ALL[i];
    }

    for (int i = 0; i < qkv_total; i++) {
        fused_qkv_b_buf[i] = TINYVIT_QKV_B_ALL[i];
    }

    for (int i = 0; i < fc1_total; i++) {
        fused_fc1_b_buf[i] = TINYVIT_FC1_B_ALL[i];
    }

    Xil_DCacheFlushRange((INTPTR)fused_norm1_w_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_norm1_b_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_qkv_b_buf, FUSED_QKV_B_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_attn_proj_b_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_norm2_w_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_norm2_b_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_fc1_b_buf, FUSED_FC1_B_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_fc2_b_buf, FUSED_NORM_BLOCK_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_attn_proj_w_buf,
        VIT_DEPTH * FUSED_ATTN_PROJ_W_FLOAT_BUF_SIZE);

    signed char (*head_w)[LIN_OUT_MAX] = (signed char (*)[LIN_OUT_MAX])LIN_STATIC_HEAD_W_ADDR;
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int c = 0; c < CLASSES; c++) {
            head_w[i][c] = TINYVIT_HEAD_W_I8[c * EMBED_DIM + i];
        }
    }
    Xil_DCacheFlushRange((INTPTR)head_w, EMBED_DIM * CLASSES * sizeof(signed char));

    prepare_gelu_lut();
    Xil_DCacheFlush();

#if DEBUG_FUSED_LAYER_COMPARE
    uart1_puts("FUSED_PARAM_DBG qkv_addr=");
    uart1_put_hex(FUSED_QKV_W_LAYER_ADDR(0));
    uart1_puts(" qkv00_bits=");
    uart1_put_hex(float_bits(((float (*)[QKV_DIM])FUSED_QKV_W_LAYER_ADDR(0))[0][0]));
    uart1_puts(" norm1w0_bits=");
    uart1_put_hex(float_bits(fused_norm1_w_buf[0]));
    uart1_puts(" fc1_addr=");
    uart1_put_hex(MLP_FUSED_FC1_W_LAYER_ADDR(0));
    uart1_puts(" fc1_00_bits=");
    uart1_put_hex(float_bits(((float (*)[EMBED_DIM])MLP_FUSED_FC1_W_LAYER_ADDR(0))[0][0]));
    uart1_puts(" attn_addr=");
    uart1_put_hex(FUSED_ATTN_PROJ_W_LAYER_ADDR(0));
    uart1_puts(" attn00_bits=");
    uart1_put_hex(float_bits(((float (*)[EMBED_DIM])FUSED_ATTN_PROJ_W_LAYER_ADDR(0))[0][0]));
    uart1_puts("\r\n");
#endif
}

static int run_gelu_lut_ip(void)
{
    u32 timeout = GELU_TIMEOUT;
    const float index_scale =
        (float)(GELU_LUT_ENTRIES - 1) / (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE);

    prepare_gelu_lut();

    Xil_DCacheFlushRange((INTPTR)gelu_x_buf, GELU_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)gelu_lut_buf, GELU_LUT_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)gelu_y_buf, GELU_Y_BUF_SIZE);

    gelu_write64(GELU_X_DATA, GELU_X_BUF_ADDR);
    gelu_write64(GELU_LUT_DATA, GELU_LUT_BUF_ADDR);
    gelu_write64(GELU_Y_DATA, GELU_Y_BUF_ADDR);
    Xil_Out32(GELU_BASE + GELU_LUT_MIN_DATA, float_bits(GELU_LUT_MIN_VALUE));
    Xil_Out32(GELU_BASE + GELU_INDEX_SCALE_DATA, float_bits(index_scale));
    Xil_Out32(GELU_BASE + GELU_AP_CTRL, 0x01U);

    while (((Xil_In32(GELU_BASE + GELU_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)gelu_y_buf, GELU_Y_BUF_SIZE);
    return 0;
}

static void clear_linear_buffers(void)
{
    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < LIN_IN_MAX; i++) {
            lin_x_buf[t][i] = 0;
        }
    }

    for (int i = 0; i < LIN_IN_MAX; i++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_w_buf[i][o] = 0;
        }
    }

    for (int o = 0; o < LIN_OUT_MAX; o++) {
        lin_bias_buf[o] = 0;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_y_buf[t][o] = 0;
        }
    }
}

static int run_linear_ip_dims(int in_dim, int out_dim)
{
    u32 timeout = LINEAR_TIMEOUT;

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_w_buf, LIN_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_W_BUF_ADDR);
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, (u32)in_dim);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, (u32)out_dim);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);
    return 0;
}

static int argmax10(void)
{
    int best = 0;

    for (int i = 1; i < CLASSES; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static int run_pair_mlp(
    const float image[EMNIST_RESOLVER_INPUTS],
    const float w1[EMNIST_RESOLVER_HIDDEN * EMNIST_RESOLVER_INPUTS],
    const float b1[EMNIST_RESOLVER_HIDDEN],
    const float w2[2 * EMNIST_RESOLVER_HIDDEN],
    const float b2[2])
{
    float hidden[EMNIST_RESOLVER_HIDDEN];
    float out0 = b2[0];
    float out1 = b2[1];

    for (int h = 0; h < EMNIST_RESOLVER_HIDDEN; h++) {
        float acc = b1[h];
        const float *row = &w1[h * EMNIST_RESOLVER_INPUTS];
        for (int i = 0; i < EMNIST_RESOLVER_INPUTS; i++) {
            acc += row[i] * image[i];
        }
        hidden[h] = gelu(acc);
        out0 += w2[h] * hidden[h];
        out1 += w2[EMNIST_RESOLVER_HIDDEN + h] * hidden[h];
    }
    return (out1 > out0) ? 1 : 0;
}

static int resolve_emnist_pair(int base_pred, const float image[UDP_IMAGE_BYTES])
{
    if (base_pred == 8 || base_pred == 9) {
        int choice = run_pair_mlp(
            image, EMNIST_IJ_W1, EMNIST_IJ_B1, EMNIST_IJ_W2, EMNIST_IJ_B2);
        return choice ? 9 : 8;
    }
    if (base_pred == 6 || base_pred == 24) {
        int choice = run_pair_mlp(
            image, EMNIST_GY_W1, EMNIST_GY_B1, EMNIST_GY_W2, EMNIST_GY_B2);
        return choice ? 24 : 6;
    }
    if (base_pred == 20 || base_pred == 21) {
        int best = 0;
        float best_distance = 3.4e38f;
        for (int p = 0; p < 2 * EMNIST_UV_PROTOTYPES_PER_CLASS; p++) {
            float distance = 0.0f;
            const float *center = &EMNIST_UV_CENTERS[p * EMNIST_RESOLVER_INPUTS];
            for (int i = 0; i < EMNIST_RESOLVER_INPUTS; i++) {
                float pixel = image[i] * VIT_INPUT_STD + VIT_INPUT_MEAN;
                float diff = pixel - center[i];
                distance += diff * diff;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best = p;
            }
        }
        return (best >= EMNIST_UV_PROTOTYPES_PER_CLASS) ? 21 : 20;
    }
    if (base_pred == 15) {
        float upper_right = 0.0f;
        float lower_right = 0.0f;
        for (int r = 0; r < 14; r++) {
            for (int c = 16; c < 28; c++) {
                upper_right += image[r * 28 + c] * VIT_INPUT_STD + VIT_INPUT_MEAN;
            }
        }
        for (int r = 14; r < 28; r++) {
            for (int c = 16; c < 28; c++) {
                lower_right += image[r * 28 + c] * VIT_INPUT_STD + VIT_INPUT_MEAN;
            }
        }
        if (lower_right > upper_right * 0.50f) {
            return 3;
        }
    }
    return base_pred;
}

static void compute_patch_tokens(int sample)
{
    int image_base = sample * 28 * 28;

    for (int e = 0; e < EMBED_DIM; e++) {
        token[0][e] = TINYVIT_CLS[e] + TINYVIT_POS[e];
    }

    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            int tok = 1 + py * 4 + px;

            for (int e = 0; e < EMBED_DIM; e++) {
                float acc = TINYVIT_PATCH_B[e];

                for (int ky = 0; ky < 7; ky++) {
                    for (int kx = 0; kx < 7; kx++) {
                        int img_idx = image_base + (py * 7 + ky) * 28 + (px * 7 + kx);
                        int w_idx = e * 49 + ky * 7 + kx;
                        acc += TINYVIT_SAMPLE_IMAGES[img_idx] * TINYVIT_PATCH_W[w_idx];
                    }
                }

                token[tok][e] = acc + TINYVIT_POS[tok * EMBED_DIM + e];
            }
        }
    }
}

static void compute_patch_tokens_from_image(const float image[UDP_IMAGE_BYTES])
{
    for (int e = 0; e < EMBED_DIM; e++) {
        token[0][e] = TINYVIT_CLS[e] + TINYVIT_POS[e];
    }

    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            int tok = 1 + py * 4 + px;

            for (int e = 0; e < EMBED_DIM; e++) {
                float acc = TINYVIT_PATCH_B[e];

                for (int ky = 0; ky < 7; ky++) {
                    for (int kx = 0; kx < 7; kx++) {
                        int img_idx = (py * 7 + ky) * 28 + (px * 7 + kx);
                        int w_idx = e * 49 + ky * 7 + kx;
                        acc += image[img_idx] * TINYVIT_PATCH_W[w_idx];
                    }
                }

                token[tok][e] = acc + TINYVIT_POS[tok * EMBED_DIM + e];
            }
        }
    }
}

static float calc_qkv_x_scale(void)
{
    float max_abs = 0.0f;

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            float v = normed[t][e];
            if (v < 0.0f) {
                v = -v;
            }
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }

    if (max_abs < 1.0e-6f) {
        max_abs = 1.0e-6f;
    }
    return max_abs / 127.0f;
}

#if USE_LEGACY_SPLIT_PL
static int run_qkv_ip(int sample)
{
    XVit_qkv_linear ip;
    u32 timeout = QKV_TIMEOUT;
    float x_scale = (sample >= 0) ? TINYVIT_SAMPLE_QKV_X_SCALES[sample] :
        calc_qkv_x_scale();
    int bias_base = (sample >= 0) ? (sample * QKV_DIM) : 0;

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            x_buf[t][e] = quantize_i8(normed[t][e], x_scale);
        }
    }

    for (int e = 0; e < EMBED_DIM; e++) {
        for (int q = 0; q < QKV_DIM; q++) {
            w_buf[e][q] = QKV_WEIGHT_INT8[e][q];
        }
    }

    for (int q = 0; q < QKV_DIM; q++) {
        bias_buf[q] = TINYVIT_SAMPLE_QKV_BIAS_I32[bias_base + q];
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int q = 0; q < QKV_DIM; q++) {
            y_buf[t][q] = 0;
        }
    }

    Xil_DCacheFlushRange((INTPTR)x_buf, X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)w_buf, W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)bias_buf, BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)y_buf, Y_BUF_SIZE);

    if (XVit_qkv_linear_Initialize(&ip, XPAR_VIT_QKV_LINEAR_0_DEVICE_ID) != XST_SUCCESS) {
        return -1;
    }

    XVit_qkv_linear_Set_x(&ip, (UINTPTR)x_buf);
    XVit_qkv_linear_Set_w(&ip, (UINTPTR)w_buf);
    XVit_qkv_linear_Set_bias(&ip, (UINTPTR)bias_buf);
    XVit_qkv_linear_Set_y(&ip, (UINTPTR)y_buf);
    XVit_qkv_linear_Start(&ip);

    while (!XVit_qkv_linear_IsDone(&ip) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -2;
    }

    Xil_DCacheInvalidateRange((INTPTR)y_buf, Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int q = 0; q < QKV_DIM; q++) {
            qkv[t][q] = (float)y_buf[t][q] * x_scale * TINYVIT_QKV_W_SCALE;
        }
    }

    return 0;
}

static int run_qkv_ip_layer(int layer)
{
    XVit_qkv_linear ip;
    u32 timeout = QKV_TIMEOUT;
    const int b_base = layer * QKV_DIM;
    float x_scale = calc_qkv_x_scale();
    float w_scale = TINYVIT_QKV_W_SCALE_ALL[layer];
    float yw_scale = x_scale * w_scale;

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            x_buf[t][e] = quantize_i8(normed[t][e], x_scale);
        }
    }

    for (int q = 0; q < QKV_DIM; q++) {
        bias_buf[q] = round_i32(TINYVIT_QKV_B_ALL[b_base + q] / yw_scale);
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int q = 0; q < QKV_DIM; q++) {
            y_buf[t][q] = 0;
        }
    }

    Xil_DCacheFlushRange((INTPTR)x_buf, X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)bias_buf, BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)y_buf, Y_BUF_SIZE);

    if (XVit_qkv_linear_Initialize(&ip, XPAR_VIT_QKV_LINEAR_0_DEVICE_ID) != XST_SUCCESS) {
        return -1;
    }

    XVit_qkv_linear_Set_x(&ip, (UINTPTR)x_buf);
    XVit_qkv_linear_Set_w(&ip, (UINTPTR)QKV_W_LAYER_ADDR(layer));
    XVit_qkv_linear_Set_bias(&ip, (UINTPTR)bias_buf);
    XVit_qkv_linear_Set_y(&ip, (UINTPTR)y_buf);
    XVit_qkv_linear_Start(&ip);

    while (!XVit_qkv_linear_IsDone(&ip) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -2;
    }

    Xil_DCacheInvalidateRange((INTPTR)y_buf, Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int q = 0; q < QKV_DIM; q++) {
            qkv[t][q] = (float)y_buf[t][q] * yw_scale;
        }
    }

    return 0;
}
#else
static int run_qkv_ip(int sample)
{
    (void)sample;
    return -99;
}

static int run_qkv_ip_layer(int layer)
{
    (void)layer;
    return -99;
}
#endif

static void compute_attention_float_ref(void)
{
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            attn_out[t][e] = 0.0f;
        }
    }

    for (int h = 0; h < HEADS; h++) {
        for (int tq = 0; tq < TOKENS; tq++) {
            float max_score = -3.4e38f;
            float sum = 0.0f;

            for (int tk = 0; tk < TOKENS; tk++) {
                float acc = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    int idx = h * HEAD_DIM + d;
                    acc += qkv[tq][idx] * qkv[tk][EMBED_DIM + idx];
                }
                float score = acc * 0.25f;
                attn_prob[tq][tk] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            for (int tk = 0; tk < TOKENS; tk++) {
                float prob = fast_expf(attn_prob[tq][tk] - max_score);
                attn_prob[tq][tk] = prob;
                sum += prob;
            }

            for (int tk = 0; tk < TOKENS; tk++) {
                attn_prob[tq][tk] = attn_prob[tq][tk] / sum;
            }
        }

        for (int tq = 0; tq < TOKENS; tq++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc = 0.0f;
                int idx = h * HEAD_DIM + d;
                for (int tk = 0; tk < TOKENS; tk++) {
                    acc += attn_prob[tq][tk] * qkv[tk][2 * EMBED_DIM + idx];
                }
                attn_out[tq][idx] = acc;
            }
        }
    }
}
static int compute_attention(void)
{
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            attn_out[t][e] = 0.0f;
        }
    }

    for (int h = 0; h < HEADS; h++) {
        float q_max = 0.0f;
        float k_max = 0.0f;
        float p_max = 0.0f;
        float v_max = 0.0f;
        float q_scale;
        float k_scale;
        float p_scale;
        float v_scale;
        float score_scale;
        float av_scale;

        for (int t = 0; t < TOKENS; t++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = h * HEAD_DIM + d;
                float qv = qkv[t][idx];
                float kv = qkv[t][EMBED_DIM + idx];
                if (qv < 0.0f) {
                    qv = -qv;
                }
                if (kv < 0.0f) {
                    kv = -kv;
                }
                if (qv > q_max) {
                    q_max = qv;
                }
                if (kv > k_max) {
                    k_max = kv;
                }
            }
        }

        q_scale = (q_max > 0.0f) ? (q_max / 127.0f) : 1.0f;
        k_scale = (k_max > 0.0f) ? (k_max / 127.0f) : 1.0f;
        score_scale = q_scale * k_scale * 0.25f;

        clear_linear_buffers();

        for (int t = 0; t < TOKENS; t++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = h * HEAD_DIM + d;
                lin_x_buf[t][d] = quantize_i8(qkv[t][idx], q_scale);
            }
        }

        for (int d = 0; d < HEAD_DIM; d++) {
            for (int tk = 0; tk < TOKENS; tk++) {
                int idx = h * HEAD_DIM + d;
                lin_w_buf[d][tk] = quantize_i8(qkv[tk][EMBED_DIM + idx], k_scale);
            }
        }

        if (run_linear_ip_dims(HEAD_DIM, TOKENS) != 0) {
            return -1;
        }

        for (int tq = 0; tq < TOKENS; tq++) {
            float max_score = -3.4e38f;
            float sum = 0.0f;

            for (int tk = 0; tk < TOKENS; tk++) {
                float score = (float)lin_y_buf[tq][tk] * score_scale;
                attn_prob[tq][tk] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            for (int tk = 0; tk < TOKENS; tk++) {
                float prob = fast_expf(attn_prob[tq][tk] - max_score);
                attn_prob[tq][tk] = prob;
                sum += prob;
            }

            for (int tk = 0; tk < TOKENS; tk++) {
                attn_prob[tq][tk] = attn_prob[tq][tk] / sum;
                if (attn_prob[tq][tk] > p_max) {
                    p_max = attn_prob[tq][tk];
                }
            }
        }

        for (int t = 0; t < TOKENS; t++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = h * HEAD_DIM + d;
                float vv = qkv[t][2 * EMBED_DIM + idx];
                if (vv < 0.0f) {
                    vv = -vv;
                }
                if (vv > v_max) {
                    v_max = vv;
                }
            }
        }

        p_scale = (p_max > 0.0f) ? (p_max / 127.0f) : 1.0f;
        v_scale = (v_max > 0.0f) ? (v_max / 127.0f) : 1.0f;
        av_scale = p_scale * v_scale;

        clear_linear_buffers();

        for (int tq = 0; tq < TOKENS; tq++) {
            for (int tk = 0; tk < TOKENS; tk++) {
                lin_x_buf[tq][tk] = quantize_i8(attn_prob[tq][tk], p_scale);
            }
        }

        for (int tk = 0; tk < TOKENS; tk++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = h * HEAD_DIM + d;
                lin_w_buf[tk][d] = quantize_i8(qkv[tk][2 * EMBED_DIM + idx], v_scale);
            }
        }

        if (run_linear_ip_dims(TOKENS, HEAD_DIM) != 0) {
            return -2;
        }

        for (int tq = 0; tq < TOKENS; tq++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                attn_out[tq][h * HEAD_DIM + d] = (float)lin_y_buf[tq][d] * av_scale;
            }
        }
    }

    return 0;
}

static int run_attn_out_proj_ip(void)
{
    u32 timeout = LINEAR_TIMEOUT;
    float x_max = max_abs_attn_out();
    float w_max = max_abs_attn_out_weight();
    float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float w_scale = (w_max > 0.0f) ? (w_max / 127.0f) : 1.0f;
    float yw_scale = x_scale * w_scale;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < LIN_IN_MAX; i++) {
            lin_x_buf[t][i] = 0;
        }
    }

    for (int i = 0; i < LIN_IN_MAX; i++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_w_buf[i][o] = 0;
        }
    }

    for (int o = 0; o < LIN_OUT_MAX; o++) {
        lin_bias_buf[o] = 0;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_y_buf[t][o] = 0;
        }
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            lin_x_buf[t][i] = quantize_i8(attn_out[t][i], x_scale);
        }
    }

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            lin_w_buf[i][o] = quantize_i8(TINYVIT_ATTN_OUT_W[o * EMBED_DIM + i], w_scale);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        lin_bias_buf[o] = round_i32(TINYVIT_ATTN_OUT_B[o] / yw_scale);
    }

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_w_buf, LIN_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_W_BUF_ADDR);
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] += (float)lin_y_buf[t][e] * yw_scale;
        }
    }

    return 0;
}

static int run_attn_out_proj_ip_layer(int layer)
{
    u32 timeout = LINEAR_TIMEOUT;
    const int b_base = layer * EMBED_DIM;
    float x_max = max_abs_attn_out();
    float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float w_scale = TINYVIT_ATTN_OUT_W_SCALE_ALL[layer];
    float yw_scale = x_scale * w_scale;

    clear_linear_buffers();

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            lin_x_buf[t][i] = quantize_i8(attn_out[t][i], x_scale);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        lin_bias_buf[o] = round_i32(TINYVIT_ATTN_OUT_B_ALL[b_base + o] / yw_scale);
    }

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_ATTN_OUT_W_LAYER_ADDR(layer));
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] += (float)lin_y_buf[t][e] * yw_scale;
        }
    }

    return 0;
}

static void compute_mlp_norm(void)
{
    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(token[t], TINYVIT_NORM2_W, TINYVIT_NORM2_B, normed[t]);
    }
}

static int MAYBE_UNUSED run_fc1_ip(void)
{
    u32 timeout = LINEAR_TIMEOUT;
    float x_max = max_abs_fc1_input();
    float w_max = max_abs_fc1_weight();
    float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float w_scale = (w_max > 0.0f) ? (w_max / 127.0f) : 1.0f;
    float yw_scale = x_scale * w_scale;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < LIN_IN_MAX; i++) {
            lin_x_buf[t][i] = 0;
        }
    }

    for (int i = 0; i < LIN_IN_MAX; i++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_w_buf[i][o] = 0;
        }
    }

    for (int o = 0; o < LIN_OUT_MAX; o++) {
        lin_bias_buf[o] = 0;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < MLP_DIM; o++) {
            lin_y_buf[t][o] = 0;
        }
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            lin_x_buf[t][i] = quantize_i8(normed[t][i], x_scale);
        }
    }

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int o = 0; o < MLP_DIM; o++) {
            lin_w_buf[i][o] = quantize_i8(TINYVIT_FC1_W[o * EMBED_DIM + i], w_scale);
        }
    }

    for (int o = 0; o < MLP_DIM; o++) {
        lin_bias_buf[o] = round_i32(TINYVIT_FC1_B[o] / yw_scale);
    }

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_w_buf, LIN_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_W_BUF_ADDR);
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, MLP_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < MLP_DIM; o++) {
            gelu_x_buf[t][o] = (float)lin_y_buf[t][o] * yw_scale;
            gelu_y_buf[t][o] = 0.0f;
        }
    }

    if (run_gelu_lut_ip() != 0) {
        return -2;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < MLP_DIM; o++) {
            mlp_hidden[t][o] = gelu_y_buf[t][o];
        }
    }

    return 0;
}

static int MAYBE_UNUSED run_fc2_ip(void)
{
    u32 timeout = LINEAR_TIMEOUT;
    float x_max = max_abs_mlp_hidden();
    float w_max = max_abs_fc2_weight();
    float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float w_scale = (w_max > 0.0f) ? (w_max / 127.0f) : 1.0f;
    float yw_scale = x_scale * w_scale;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < LIN_IN_MAX; i++) {
            lin_x_buf[t][i] = 0;
        }
    }

    for (int i = 0; i < LIN_IN_MAX; i++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_w_buf[i][o] = 0;
        }
    }

    for (int o = 0; o < LIN_OUT_MAX; o++) {
        lin_bias_buf[o] = 0;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_y_buf[t][o] = 0;
        }
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < MLP_DIM; i++) {
            lin_x_buf[t][i] = quantize_i8(mlp_hidden[t][i], x_scale);
        }
    }

    for (int i = 0; i < MLP_DIM; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            lin_w_buf[i][o] = quantize_i8(TINYVIT_FC2_W[o * MLP_DIM + i], w_scale);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        lin_bias_buf[o] = round_i32(TINYVIT_FC2_B[o] / yw_scale);
    }

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_w_buf, LIN_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_W_BUF_ADDR);
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, MLP_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            token[t][o] += (float)lin_y_buf[t][o] * yw_scale;
        }
    }

    return 0;
}

static int run_mlp_fused_ip(void)
{
    u32 timeout = MLP_FUSED_TIMEOUT;
    float fc1_x_max = max_abs_fc1_input();
    float fc1_w_max = max_abs_fc1_weight();
    float fc2_w_max = max_abs_fc2_weight();
    float fc1_x_scale = (fc1_x_max > 0.0f) ? (fc1_x_max / 127.0f) : 1.0f;
    float fc1_w_scale = (fc1_w_max > 0.0f) ? (fc1_w_max / 127.0f) : 1.0f;
    float fc1_scale = fc1_x_scale * fc1_w_scale;
    float hidden_inv_scale = MLP_FUSED_HIDDEN_INV_SCALE;
    float hidden_scale = 1.0f / hidden_inv_scale;
    float fc2_w_scale = (fc2_w_max > 0.0f) ? (fc2_w_max / 127.0f) : 1.0f;
    float fc2_output_scale = hidden_scale * fc2_w_scale;
    float lut_index_scale =
        (float)(GELU_LUT_ENTRIES - 1) / (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE);

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            mlp_fused_x_buf[t][i] = quantize_i8(normed[t][i], fc1_x_scale);
            mlp_fused_y_buf[t][i] = 0.0f;
        }
    }

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int o = 0; o < MLP_DIM; o++) {
            mlp_fused_fc1_w_buf[i][o] =
                quantize_i8(TINYVIT_FC1_W[o * EMBED_DIM + i], fc1_w_scale);
        }
    }

    for (int o = 0; o < MLP_DIM; o++) {
        mlp_fused_fc1_b_buf[o] = round_i32(TINYVIT_FC1_B[o] / fc1_scale);
    }

    for (int i = 0; i < GELU_LUT_ENTRIES; i++) {
        float x = GELU_LUT_MIN_VALUE +
            (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE) *
            (float)i / (float)(GELU_LUT_ENTRIES - 1);
        mlp_fused_lut_buf[i] = gelu(x);
    }

    for (int i = 0; i < MLP_DIM; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            mlp_fused_fc2_w_buf[i][o] =
                quantize_i8(TINYVIT_FC2_W[o * MLP_DIM + i], fc2_w_scale);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        mlp_fused_fc2_b_buf[o] = round_i32(TINYVIT_FC2_B[o] / fc2_output_scale);
    }

    Xil_DCacheFlushRange((INTPTR)mlp_fused_x_buf, MLP_FUSED_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc1_w_buf, MLP_FUSED_FC1_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc1_b_buf, MLP_FUSED_FC1_B_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_lut_buf, MLP_FUSED_LUT_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc2_w_buf, MLP_FUSED_FC2_W_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc2_b_buf, MLP_FUSED_FC2_B_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_y_buf, MLP_FUSED_Y_BUF_SIZE);

    mlp_fused_write64(MLP_FUSED_X_DATA, MLP_FUSED_X_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC1_W_DATA, MLP_FUSED_FC1_W_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC1_B_DATA, MLP_FUSED_FC1_B_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_GELU_LUT_DATA, MLP_FUSED_LUT_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC2_W_DATA, MLP_FUSED_FC2_W_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC2_B_DATA, MLP_FUSED_FC2_B_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_Y_DATA, MLP_FUSED_Y_BUF_ADDR);
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_FC1_SCALE_DATA, float_bits(fc1_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_HIDDEN_INV_SCALE_DATA, float_bits(hidden_inv_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_FC2_OUTPUT_SCALE_DATA, float_bits(fc2_output_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_LUT_MIN_DATA, float_bits(GELU_LUT_MIN_VALUE));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_LUT_INDEX_SCALE_DATA, float_bits(lut_index_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_AP_CTRL, 0x01U);

    while (((Xil_In32(MLP_FUSED_BASE + MLP_FUSED_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)mlp_fused_y_buf, MLP_FUSED_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            token[t][o] += mlp_fused_y_buf[t][o];
        }
    }

    return 0;
}

static void compute_mlp_norm_layer(int layer)
{
    const int base = layer * EMBED_DIM;

    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(
            token[t],
            &TINYVIT_NORM2_W_ALL[base],
            &TINYVIT_NORM2_B_ALL[base],
            normed[t]);
    }
}

static int run_mlp_fused_ip_layer(int layer)
{
    u32 timeout = MLP_FUSED_TIMEOUT;
    const int fc1_b_base = layer * MLP_DIM;
    const int fc2_b_base = layer * EMBED_DIM;
    float fc1_x_max = max_abs_fc1_input();

    float fc1_x_scale = (fc1_x_max > 0.0f) ? (fc1_x_max / 127.0f) : 1.0f;
    float fc1_w_scale = TINYVIT_FC1_W_SCALE_ALL[layer];
    float fc1_scale = fc1_x_scale * fc1_w_scale;
    float hidden_inv_scale = MLP_FUSED_HIDDEN_INV_SCALE;
    float hidden_scale = 1.0f / hidden_inv_scale;
    float fc2_w_scale = TINYVIT_FC2_W_SCALE_ALL[layer];
    float fc2_output_scale = hidden_scale * fc2_w_scale;
    float lut_index_scale =
        (float)(GELU_LUT_ENTRIES - 1) / (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE);

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            mlp_fused_x_buf[t][i] = quantize_i8(normed[t][i], fc1_x_scale);
            mlp_fused_y_buf[t][i] = 0.0f;
        }
    }

    for (int h = 0; h < MLP_DIM; h++) {
        mlp_fused_fc1_b_buf[h] = round_i32(TINYVIT_FC1_B_ALL[fc1_b_base + h] / fc1_scale);
    }

    for (int e = 0; e < EMBED_DIM; e++) {
        mlp_fused_fc2_b_buf[e] = round_i32(TINYVIT_FC2_B_ALL[fc2_b_base + e] / fc2_output_scale);
    }

    Xil_DCacheFlushRange((INTPTR)mlp_fused_x_buf, MLP_FUSED_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc1_b_buf, MLP_FUSED_FC1_B_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_fc2_b_buf, MLP_FUSED_FC2_B_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)mlp_fused_y_buf, MLP_FUSED_Y_BUF_SIZE);

    mlp_fused_write64(MLP_FUSED_X_DATA, MLP_FUSED_X_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC1_W_DATA, MLP_FUSED_FC1_W_LAYER_ADDR(layer));
    mlp_fused_write64(MLP_FUSED_FC1_B_DATA, MLP_FUSED_FC1_B_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_GELU_LUT_DATA, MLP_FUSED_LUT_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_FC2_W_DATA, MLP_FUSED_FC2_W_LAYER_ADDR(layer));
    mlp_fused_write64(MLP_FUSED_FC2_B_DATA, MLP_FUSED_FC2_B_BUF_ADDR);
    mlp_fused_write64(MLP_FUSED_Y_DATA, MLP_FUSED_Y_BUF_ADDR);
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_FC1_SCALE_DATA, float_bits(fc1_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_HIDDEN_INV_SCALE_DATA, float_bits(hidden_inv_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_FC2_OUTPUT_SCALE_DATA, float_bits(fc2_output_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_LUT_MIN_DATA, float_bits(GELU_LUT_MIN_VALUE));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_LUT_INDEX_SCALE_DATA, float_bits(lut_index_scale));
    Xil_Out32(MLP_FUSED_BASE + MLP_FUSED_AP_CTRL, 0x01U);

    while (((Xil_In32(MLP_FUSED_BASE + MLP_FUSED_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)mlp_fused_y_buf, MLP_FUSED_Y_BUF_SIZE);

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] += mlp_fused_y_buf[t][e];
        }
    }

    return 0;
}

static void compute_norm1_layer(int layer)
{
    const int base = layer * EMBED_DIM;

    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(
            token[t],
            &TINYVIT_NORM1_W_ALL[base],
            &TINYVIT_NORM1_B_ALL[base],
            normed[t]);
    }
}

static void compute_qkv_float_layer(int layer)
{
    const int w_base = layer * QKV_DIM * EMBED_DIM;
    const int b_base = layer * QKV_DIM;

    for (int t = 0; t < TOKENS; t++) {
        for (int q = 0; q < QKV_DIM; q++) {
            float acc = TINYVIT_QKV_B_ALL[b_base + q];
            for (int e = 0; e < EMBED_DIM; e++) {
                acc += normed[t][e] *
                    TINYVIT_QKV_W_ALL[w_base + q * EMBED_DIM + e];
            }
            qkv[t][q] = acc;
        }
    }
}

static void run_attn_out_proj_cpu_layer(int layer)
{
    const int w_base = layer * EMBED_DIM * EMBED_DIM;
    const int b_base = layer * EMBED_DIM;

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            float acc = TINYVIT_ATTN_OUT_B_ALL[b_base + o];
            for (int i = 0; i < EMBED_DIM; i++) {
                acc += attn_out[t][i] *
                    TINYVIT_ATTN_OUT_W_ALL[w_base + o * EMBED_DIM + i];
            }
            token[t][o] += acc;
        }
    }
}

static void run_mlp_cpu_layer(int layer)
{
    const int norm_base = layer * EMBED_DIM;
    const int fc1_w_base = layer * MLP_DIM * EMBED_DIM;
    const int fc1_b_base = layer * MLP_DIM;
    const int fc2_w_base = layer * EMBED_DIM * MLP_DIM;
    const int fc2_b_base = layer * EMBED_DIM;

    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(
            token[t],
            &TINYVIT_NORM2_W_ALL[norm_base],
            &TINYVIT_NORM2_B_ALL[norm_base],
            normed[t]);
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int h = 0; h < MLP_DIM; h++) {
            float acc = TINYVIT_FC1_B_ALL[fc1_b_base + h];
            for (int e = 0; e < EMBED_DIM; e++) {
                acc += normed[t][e] *
                    TINYVIT_FC1_W_ALL[fc1_w_base + h * EMBED_DIM + e];
            }
            mlp_hidden[t][h] = gelu(acc);
        }
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            float acc = TINYVIT_FC2_B_ALL[fc2_b_base + e];
            for (int h = 0; h < MLP_DIM; h++) {
                acc += mlp_hidden[t][h] *
                    TINYVIT_FC2_W_ALL[fc2_w_base + e * MLP_DIM + h];
            }
            token[t][e] += acc;
        }
    }
}

static int run_transformer_layer_deepwide(int layer)
{
    int status;

    compute_norm1_layer(layer);
    compute_qkv_float_layer(layer);

    status = compute_attention();
    if (status != 0) {
        return status;
    }

    run_attn_out_proj_cpu_layer(layer);
    run_mlp_cpu_layer(layer);
    return 0;
}

static int run_transformer_stack_deepwide(void)
{
    int status;

    for (int layer = 0; layer < VIT_DEPTH; layer++) {
        status = run_transformer_layer_deepwide(layer);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

static int run_transformer_layer_fused_ip(int layer)
{
    u32 timeout = TRANSFORMER_FUSED_TIMEOUT;
    const float lut_index_scale =
        (float)(GELU_LUT_ENTRIES - 1) / (GELU_LUT_MAX_VALUE - GELU_LUT_MIN_VALUE);

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            fused_token_a_buf[t][e] = token[t][e];
            fused_token_b_buf[t][e] = 0.0f;
        }
    }

    Xil_DCacheFlushRange((INTPTR)fused_token_a_buf, FUSED_TOKEN_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)fused_token_b_buf, FUSED_TOKEN_BUF_SIZE);
    Xil_DCacheFlush();

#if DEBUG_FUSED_LAYER_COMPARE
    if (layer == 0) {
        uart1_puts("FUSED_PRE_IP token00=");
        uart1_put_hex(float_bits(fused_token_a_buf[0][0]));
        uart1_puts(" norm1w0=");
        uart1_put_hex(float_bits(((float *)FUSED_LAYER_NORM_ADDR(FUSED_NORM1_W_ADDR, layer))[0]));
        uart1_puts(" qkv00=");
        uart1_put_hex(float_bits(((float (*)[QKV_DIM])FUSED_QKV_W_LAYER_ADDR(layer))[0][0]));
        uart1_puts(" qkv01=");
        uart1_put_hex(float_bits(((float (*)[QKV_DIM])FUSED_QKV_W_LAYER_ADDR(layer))[0][1]));
        uart1_puts(" attn00=");
        uart1_put_hex(float_bits(((float (*)[EMBED_DIM])FUSED_ATTN_PROJ_W_LAYER_ADDR(layer))[0][0]));
        uart1_puts(" fc100=");
        uart1_put_hex(float_bits(((float (*)[EMBED_DIM])MLP_FUSED_FC1_W_LAYER_ADDR(layer))[0][0]));
        uart1_puts("\r\n");
    }
#endif

    transformer_fused_write64(TRANSFORMER_FUSED_TOKEN_IN_DATA, FUSED_TOKEN_A_BUF_ADDR);
    transformer_fused_write64(TRANSFORMER_FUSED_NORM1_W_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_NORM1_W_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_NORM1_B_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_NORM1_B_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_QKV_W_DATA, FUSED_QKV_W_LAYER_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_QKV_B_DATA, FUSED_LAYER_QKV_B_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_ATTN_PROJ_W_DATA,
        FUSED_ATTN_PROJ_W_LAYER_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_ATTN_PROJ_B_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_ATTN_PROJ_B_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_NORM2_W_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_NORM2_W_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_NORM2_B_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_NORM2_B_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_FC1_W_DATA, MLP_FUSED_FC1_W_LAYER_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_FC1_B_DATA, FUSED_LAYER_FC1_B_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_GELU_LUT_DATA, MLP_FUSED_LUT_BUF_ADDR);
    transformer_fused_write64(TRANSFORMER_FUSED_FC2_W_DATA, MLP_FUSED_FC2_W_LAYER_ADDR(layer));
    transformer_fused_write64(TRANSFORMER_FUSED_FC2_B_DATA,
        FUSED_LAYER_NORM_ADDR(FUSED_FC2_B_ADDR, layer));
    transformer_fused_write64(TRANSFORMER_FUSED_TOKEN_OUT_DATA, FUSED_TOKEN_B_BUF_ADDR);

    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_QKV_W_SCALE_DATA,
        float_bits(TINYVIT_QKV_W_SCALE_ALL[layer]));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_ATTN_PROJ_W_SCALE_DATA,
        float_bits(TINYVIT_ATTN_OUT_W_SCALE_ALL[layer]));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_FC1_W_SCALE_DATA,
        float_bits(TINYVIT_FC1_W_SCALE_ALL[layer]));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_FC2_W_SCALE_DATA,
        float_bits(TINYVIT_FC2_W_SCALE_ALL[layer]));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_HIDDEN_INV_SCALE_DATA,
        float_bits((DEBUG_FUSED_IP_STAGE > 0) ? (-(float)DEBUG_FUSED_IP_STAGE) : MLP_FUSED_HIDDEN_INV_SCALE));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_LUT_MIN_DATA,
        float_bits(GELU_LUT_MIN_VALUE));
    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_LUT_INDEX_SCALE_DATA,
        float_bits(lut_index_scale));

    Xil_Out32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_AP_CTRL, 0x01U);
    while (((Xil_In32(TRANSFORMER_FUSED_BASE + TRANSFORMER_FUSED_AP_CTRL) & 0x02U) == 0U) &&
           timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidate();
    Xil_DCacheInvalidateRange((INTPTR)fused_token_b_buf, FUSED_TOKEN_BUF_SIZE);
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] = fused_token_b_buf[t][e];
        }
    }

    return 0;
}

#if DEBUG_FUSED_LAYER_COMPARE
static int run_transformer_debug_ref_stage(int layer)
{
#if DEBUG_FUSED_IP_STAGE == 1
    const int base = layer * EMBED_DIM;
    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(
            token[t],
            &TINYVIT_NORM1_W_ALL[base],
            &TINYVIT_NORM1_B_ALL[base],
            normed[t]);
    }
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] = normed[t][e];
        }
    }
    return 0;
#elif DEBUG_FUSED_IP_STAGE == 2
    compute_norm1_layer(layer);
    compute_qkv_float_layer(layer);
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] = qkv[t][e];
        }
    }
    return 0;
#elif DEBUG_FUSED_IP_STAGE == 3
    compute_norm1_layer(layer);
    compute_qkv_float_layer(layer);
    compute_attention_float_ref();
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] = attn_out[t][e];
        }
    }
    return 0;
#elif DEBUG_FUSED_IP_STAGE == 4
    compute_norm1_layer(layer);
    compute_qkv_float_layer(layer);
    compute_attention_float_ref();
    run_attn_out_proj_cpu_layer(layer);
    return 0;
#else
    return run_transformer_layer_deepwide(layer);
#endif
}

static int run_transformer_stack_fused_debug_compare(void)
{
    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            fused_debug_ref_token[t][e] = token[t][e];
            fused_debug_pl_token[t][e] = token[t][e];
        }
    }

    for (int layer = 0; layer < VIT_DEPTH; layer++) {
        for (int t = 0; t < TOKENS; t++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                token[t][e] = fused_debug_ref_token[t][e];
            }
        }
        if (run_transformer_debug_ref_stage(layer) != 0) {
            return -10 - layer;
        }
        for (int t = 0; t < TOKENS; t++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                fused_debug_ref_token[t][e] = token[t][e];
            }
        }

        for (int t = 0; t < TOKENS; t++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                token[t][e] = fused_debug_pl_token[t][e];
            }
        }
        if (run_transformer_layer_fused_ip(layer) != 0) {
            return -20 - layer;
        }
        for (int t = 0; t < TOKENS; t++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                fused_debug_pl_token[t][e] = token[t][e];
            }
        }

        float max_diff = 0.0f;
        u32 nan_count = 0U;
        for (int t = 0; t < TOKENS; t++) {
            for (int e = 0; e < EMBED_DIM; e++) {
                float ref_v = fused_debug_ref_token[t][e];
                float pl_v = fused_debug_pl_token[t][e];
                float d = ref_v - pl_v;
                if ((ref_v != ref_v) || (pl_v != pl_v) ||
                    (ref_v > 1.0e20f) || (ref_v < -1.0e20f) ||
                    (pl_v > 1.0e20f) || (pl_v < -1.0e20f)) {
                    nan_count++;
                    continue;
                }
                if (d < 0.0f) {
                    d = -d;
                }
                if (d > max_diff) {
                    max_diff = d;
                }
            }
        }

        uart1_puts("FUSED_DBG layer=");
        uart1_put_dec_u32((u32)layer);
        uart1_puts(" max_diff_milli=");
        uart1_put_dec_u32((u32)(max_diff * 1000.0f));
        uart1_puts(" nan=");
        uart1_put_dec_u32(nan_count);
        uart1_puts(" ref_cls0_milli=");
        uart1_put_hex((u32)((int)(fused_debug_ref_token[0][0] * 1000.0f)));
        uart1_puts(" pl_cls0_milli=");
        uart1_put_hex((u32)((int)(fused_debug_pl_token[0][0] * 1000.0f)));
        uart1_puts("\r\n");
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int e = 0; e < EMBED_DIM; e++) {
            token[t][e] = fused_debug_pl_token[t][e];
        }
    }
    return 0;
}
#endif

static int run_transformer_stack_fused_ip(void)
{
    int status;

    prof_norm_ms = 0U;
    prof_qkv_ms = 0U;
    prof_attn_ms = 0U;
    prof_proj_ms = 0U;
    prof_mlp_ms = 0U;

#if DEBUG_FUSED_LAYER_COMPARE
    return run_transformer_stack_fused_debug_compare();
#endif

    for (int layer = 0; layer < VIT_DEPTH; layer++) {
        status = run_transformer_layer_fused_ip(layer);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

#if USE_LEGACY_SPLIT_PL
static int run_transformer_layer_pl_accel(int layer)
{
    int status;
    XTime t0;
    XTime t1;

    XTime_GetTime(&t0);

    compute_norm1_layer(layer);
    XTime_GetTime(&t1);
    prof_norm_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);
    t0 = t1;

    status = run_qkv_ip_layer(layer);
    if (status != 0) {
        return status;
    }
    XTime_GetTime(&t1);
    prof_qkv_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);
    t0 = t1;

    status = compute_attention();
    if (status != 0) {
        return status;
    }
    XTime_GetTime(&t1);
    prof_attn_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);
    t0 = t1;

    status = run_attn_out_proj_ip_layer(layer);
    if (status != 0) {
        return status;
    }
    XTime_GetTime(&t1);
    prof_proj_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);
    t0 = t1;

    compute_mlp_norm_layer(layer);
    XTime_GetTime(&t1);
    prof_norm_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);
    t0 = t1;

    status = run_mlp_fused_ip_layer(layer);
    if (status != 0) {
        return status;
    }
    XTime_GetTime(&t1);
    prof_mlp_ms += (u32)(((t1 - t0) * 1000ULL) / COUNTS_PER_SECOND);

    return 0;
}

static int run_transformer_stack_pl_accel(void)
{
    int status;

    prof_norm_ms = 0U;
    prof_qkv_ms = 0U;
    prof_attn_ms = 0U;
    prof_proj_ms = 0U;
    prof_mlp_ms = 0U;

    for (int layer = 0; layer < VIT_DEPTH; layer++) {
        status = run_transformer_layer_pl_accel(layer);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}
#else
static int run_transformer_stack_pl_accel(void)
{
    return -99;
}
#endif

static void compute_final_norm(void)
{
    layer_norm_64(token[0], TINYVIT_FINAL_NORM_W, TINYVIT_FINAL_NORM_B, normed[0]);
}

static int run_head_ip(void)
{
    for (int c = 0; c < CLASSES; c++) {
        float acc = TINYVIT_HEAD_B[c];
        for (int i = 0; i < EMBED_DIM; i++) {
            acc += normed[0][i] * TINYVIT_HEAD_W[c * EMBED_DIM + i];
        }
        logits[c] = acc;
    }
    return 0;

#if USE_LEGACY_SPLIT_PL
    u32 timeout = LINEAR_TIMEOUT;
    float x_max = max_abs_head_input();
    float x_scale = (x_max > 0.0f) ? (x_max / 127.0f) : 1.0f;
    float w_scale = TINYVIT_HEAD_W_SCALE[0];
    float yw_scale = x_scale * w_scale;

    for (int t = 0; t < TOKENS; t++) {
        for (int i = 0; i < LIN_IN_MAX; i++) {
            lin_x_buf[t][i] = 0;
        }
    }

    for (int o = 0; o < LIN_OUT_MAX; o++) {
        lin_bias_buf[o] = 0;
    }

    for (int t = 0; t < TOKENS; t++) {
        for (int o = 0; o < LIN_OUT_MAX; o++) {
            lin_y_buf[t][o] = 0;
        }
    }

    for (int i = 0; i < EMBED_DIM; i++) {
        lin_x_buf[0][i] = quantize_i8(normed[0][i], x_scale);
    }

    for (int c = 0; c < CLASSES; c++) {
        lin_bias_buf[c] = round_i32(TINYVIT_HEAD_B[c] / yw_scale);
    }

    Xil_DCacheFlushRange((INTPTR)lin_x_buf, LIN_X_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_bias_buf, LIN_BIAS_BUF_SIZE);
    Xil_DCacheFlushRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    linear_write64(LINEAR_X_DATA, LIN_X_BUF_ADDR);
    linear_write64(LINEAR_W_DATA, LIN_STATIC_HEAD_W_ADDR);
    linear_write64(LINEAR_BIAS_DATA, LIN_BIAS_BUF_ADDR);
    linear_write64(LINEAR_Y_DATA, LIN_Y_BUF_ADDR);
    Xil_Out32(LINEAR_BASE + LINEAR_IN_DIM_DATA, EMBED_DIM);
    Xil_Out32(LINEAR_BASE + LINEAR_OUT_DIM_DATA, CLASSES);
    Xil_Out32(LINEAR_BASE + LINEAR_AP_CTRL, 0x01U);

    while (((Xil_In32(LINEAR_BASE + LINEAR_AP_CTRL) & 0x02U) == 0U) && timeout > 0U) {
        timeout--;
    }

    if (timeout == 0U) {
        return -1;
    }

    Xil_DCacheInvalidateRange((INTPTR)lin_y_buf, LIN_Y_BUF_SIZE);

    for (int c = 0; c < CLASSES; c++) {
        logits[c] = (float)lin_y_buf[0][c] * yw_scale;
    }

    return 0;
#endif
}

static int MAYBE_UNUSED run_one_sample(int sample, int *label_correct)
{
    int qkv_status;
    int pred;
    int base_pred;
    int label = TINYVIT_SAMPLE_LABELS[sample];
    int pytorch_pred = TINYVIT_SAMPLE_PYTORCH_PREDS[sample];

    uart1_puts("sample=");
    uart1_put_hex((u32)sample);
    uart1_puts(" label=");
    uart1_put_hex((u32)label);
    uart1_puts(" pytorch_pred=");
    uart1_put_hex((u32)pytorch_pred);
    uart1_puts("\r\n");

    compute_patch_tokens(sample);

    if (VIT_DEPTH > 1 && MLP_DIM <= 256) {
        qkv_status = run_transformer_stack_fused_ip();
        if (qkv_status != 0) {
            uart1_puts("FUSED_LAYER_STACK_FAIL sample=");
            uart1_put_hex((u32)sample);
            uart1_puts(" status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return 0;
        }
        compute_final_norm();
        qkv_status = run_head_ip();
        if (qkv_status != 0) {
            uart1_puts("FUSED_LAYER_HEAD_FAIL sample=");
            uart1_put_hex((u32)sample);
            uart1_puts(" status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return 0;
        }
        base_pred = argmax10();
        pred = resolve_emnist_pair(
            base_pred, &TINYVIT_SAMPLE_IMAGES[sample * UDP_IMAGE_BYTES]);

        uart1_puts("result sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" pred=");
        uart1_put_hex((u32)pred);
        uart1_puts(" base_pred=");
        uart1_put_hex((u32)base_pred);
        uart1_puts(" label=");
        uart1_put_hex((u32)label);
        uart1_puts(" pytorch_pred=");
        uart1_put_hex((u32)pytorch_pred);
        uart1_puts(" logit_pred_milli=");
        uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
        uart1_puts("\r\n");

        *label_correct = (pred == label);
        return base_pred == pytorch_pred;
    }

    if (VIT_DEPTH > 1 || MLP_DIM > 128) {
        qkv_status = run_transformer_stack_deepwide();
        if (qkv_status != 0) {
            uart1_puts("DEEPWIDE_STACK_FAIL sample=");
            uart1_put_hex((u32)sample);
            uart1_puts(" status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return 0;
        }
        compute_final_norm();
        qkv_status = run_head_ip();
        if (qkv_status != 0) {
            uart1_puts("DEEPWIDE_HEAD_FAIL sample=");
            uart1_put_hex((u32)sample);
            uart1_puts(" status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return 0;
        }
        base_pred = argmax10();
        pred = resolve_emnist_pair(
            base_pred, &TINYVIT_SAMPLE_IMAGES[sample * UDP_IMAGE_BYTES]);

        uart1_puts("result sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" pred=");
        uart1_put_hex((u32)pred);
        uart1_puts(" base_pred=");
        uart1_put_hex((u32)base_pred);
        uart1_puts(" label=");
        uart1_put_hex((u32)label);
        uart1_puts(" pytorch_pred=");
        uart1_put_hex((u32)pytorch_pred);
        uart1_puts(" logit_pred_milli=");
        uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
        uart1_puts("\r\n");

        *label_correct = (pred == label);
        return base_pred == pytorch_pred;
    }

    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(token[t], TINYVIT_NORM1_W, TINYVIT_NORM1_B, normed[t]);
    }

    qkv_status = run_qkv_ip(sample);
    if (qkv_status != 0) {
        uart1_puts("PL_QKV_FAIL sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return 0;
    }

    qkv_status = compute_attention();
    if (qkv_status != 0) {
        uart1_puts("PL_ATTN_MATMUL_FAIL sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return 0;
    }
    qkv_status = run_attn_out_proj_ip();
    if (qkv_status != 0) {
        uart1_puts("PL_ATTN_OUT_FAIL sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return 0;
    }
    compute_mlp_norm();
    qkv_status = run_mlp_fused_ip();
    if (qkv_status != 0) {
        uart1_puts("PL_MLP_FUSED_FAIL sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return 0;
    }
    compute_final_norm();
    qkv_status = run_head_ip();
    if (qkv_status != 0) {
        uart1_puts("PL_HEAD_FAIL sample=");
        uart1_put_hex((u32)sample);
        uart1_puts(" status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return 0;
    }
    base_pred = argmax10();
    pred = resolve_emnist_pair(
        base_pred, &TINYVIT_SAMPLE_IMAGES[sample * UDP_IMAGE_BYTES]);

    uart1_puts("result sample=");
    uart1_put_hex((u32)sample);
    uart1_puts(" pred=");
    uart1_put_hex((u32)pred);
    uart1_puts(" base_pred=");
    uart1_put_hex((u32)base_pred);
    uart1_puts(" label=");
    uart1_put_hex((u32)label);
    uart1_puts(" pytorch_pred=");
    uart1_put_hex((u32)pytorch_pred);
    uart1_puts(" logit_pred_milli=");
    uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
    uart1_puts("\r\n");

    *label_correct = (pred == label);
    return base_pred == pytorch_pred;
}

static int run_uploaded_image(void)
{
    int qkv_status;
    int pred;
    int base_pred;
    XTime infer_start;
    XTime patch_done;
    XTime stack_done;
    XTime head_done;
    XTime infer_end;

    uart1_puts("UDP_INFER_BEGIN len=");
    uart1_put_dec_u32((u32)udp_image_len);
    uart1_puts("\r\n");

    XTime_GetTime(&infer_start);

    for (int i = 0; i < UDP_IMAGE_BYTES; i++) {
        float pixel = (float)udp_image_raw[i] / 255.0f;
        udp_image_float[i] = (pixel - VIT_INPUT_MEAN) / VIT_INPUT_STD;
    }

    compute_patch_tokens_from_image(udp_image_float);
    XTime_GetTime(&patch_done);

    if (VIT_DEPTH > 1 && MLP_DIM <= 256) {
        qkv_status = run_transformer_stack_fused_ip();
        if (qkv_status != 0) {
            uart1_puts("UDP_FUSED_LAYER_STACK_FAIL status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return -1;
        }
        XTime_GetTime(&stack_done);

        compute_final_norm();

        qkv_status = run_head_ip();
        if (qkv_status != 0) {
            uart1_puts("UDP_FUSED_LAYER_HEAD_FAIL status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return -2;
        }
        XTime_GetTime(&head_done);

        base_pred = argmax10();
        pred = resolve_emnist_pair(base_pred, udp_image_float);
        XTime_GetTime(&infer_end);
        last_infer_ms = (u32)(((infer_end - infer_start) * 1000ULL) / COUNTS_PER_SECOND);
        last_patch_ms = (u32)(((patch_done - infer_start) * 1000ULL) / COUNTS_PER_SECOND);
        last_stack_ms = (u32)(((stack_done - patch_done) * 1000ULL) / COUNTS_PER_SECOND);
        last_head_ms = (u32)(((head_done - stack_done) * 1000ULL) / COUNTS_PER_SECOND);

        uart1_puts("UDP_RESULT pred=");
        uart1_put_hex((u32)pred);
        uart1_puts(" base_pred=");
        uart1_put_hex((u32)base_pred);
        uart1_puts(" path=fused_transformer_pl");
        uart1_puts(" depth=");
        uart1_put_dec_u32((u32)VIT_DEPTH);
        uart1_puts(" embed=");
        uart1_put_dec_u32((u32)EMBED_DIM);
        uart1_puts(" mlp=");
        uart1_put_dec_u32((u32)MLP_DIM);
        uart1_puts(" classes=");
        uart1_put_dec_u32((u32)CLASSES);
        uart1_puts(" dataset=" TINYVIT_DATASET_NAME);
        uart1_puts(" logit_pred_milli=");
        uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
        uart1_puts(" infer_ms=");
        uart1_put_dec_u32(last_infer_ms);
        uart1_puts(" patch_ms=");
        uart1_put_dec_u32(last_patch_ms);
        uart1_puts(" pl_stack_ms=");
        uart1_put_dec_u32(last_stack_ms);
        uart1_puts(" head_ms=");
        uart1_put_dec_u32(last_head_ms);
        uart1_puts("\r\n");

        return pred;
    }

    if (VIT_DEPTH > 1 || MLP_DIM > 128) {
        qkv_status = run_transformer_stack_deepwide();
        if (qkv_status != 0) {
            uart1_puts("UDP_DEEPWIDE_STACK_FAIL status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return -1;
        }

        compute_final_norm();

        qkv_status = run_head_ip();
        if (qkv_status != 0) {
            uart1_puts("UDP_DEEPWIDE_HEAD_FAIL status=");
            uart1_put_hex((u32)qkv_status);
            uart1_puts("\r\n");
            return -2;
        }

        base_pred = argmax10();
        pred = resolve_emnist_pair(base_pred, udp_image_float);
        XTime_GetTime(&infer_end);
        last_infer_ms = (u32)(((infer_end - infer_start) * 1000ULL) / COUNTS_PER_SECOND);

        uart1_puts("UDP_RESULT pred=");
        uart1_put_hex((u32)pred);
        uart1_puts(" base_pred=");
        uart1_put_hex((u32)base_pred);
        uart1_puts(" logit_pred_milli=");
        uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
        uart1_puts(" infer_ms=");
        uart1_put_dec_u32(last_infer_ms);
        uart1_puts("\r\n");

        return pred;
    }

    for (int t = 0; t < TOKENS; t++) {
        layer_norm_64(token[t], TINYVIT_NORM1_W, TINYVIT_NORM1_B, normed[t]);
    }

    qkv_status = run_qkv_ip(-1);
    if (qkv_status != 0) {
        uart1_puts("UDP_PL_QKV_FAIL status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return -1;
    }

    qkv_status = compute_attention();
    if (qkv_status != 0) {
        uart1_puts("UDP_PL_ATTN_MATMUL_FAIL status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return -2;
    }

    qkv_status = run_attn_out_proj_ip();
    if (qkv_status != 0) {
        uart1_puts("UDP_PL_ATTN_OUT_FAIL status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return -3;
    }

    compute_mlp_norm();

    qkv_status = run_mlp_fused_ip();
    if (qkv_status != 0) {
        uart1_puts("UDP_PL_MLP_FUSED_FAIL status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return -4;
    }

    compute_final_norm();

    qkv_status = run_head_ip();
    if (qkv_status != 0) {
        uart1_puts("UDP_PL_HEAD_FAIL status=");
        uart1_put_hex((u32)qkv_status);
        uart1_puts("\r\n");
        return -5;
    }

    base_pred = argmax10();
    pred = resolve_emnist_pair(base_pred, udp_image_float);
    XTime_GetTime(&infer_end);
    last_infer_ms = (u32)(((infer_end - infer_start) * 1000ULL) / COUNTS_PER_SECOND);

    uart1_puts("UDP_RESULT pred=");
    uart1_put_hex((u32)pred);
    uart1_puts(" base_pred=");
    uart1_put_hex((u32)base_pred);
    uart1_puts(" logit_pred_milli=");
    uart1_put_hex((u32)((int)(logits[pred] * 1000.0f)));
    uart1_puts(" infer_ms=");
    uart1_put_dec_u32(last_infer_ms);
    uart1_puts("\r\n");

    return pred;
}

int main()
{
    boot_marker = 0x11111111U;
    uart1_init_direct();
    uart1_puts("\r\nFUSED_BOOT_EARLY build=2026-06-19 path=fused_transformer_pl\r\n");
    uart1_puts("FUSED_BOOT_NET ip=" DEFAULT_IP_ADDRESS " port=5001 uart=UART1\r\n");
    uart1_puts("FUSED_BOOT_MODEL dataset=" TINYVIT_DATASET_NAME " classes=");
    uart1_put_dec_u32((u32)CLASSES);
    uart1_puts(" depth=");
    uart1_put_dec_u32((u32)VIT_DEPTH);
    uart1_puts(" embed=");
    uart1_put_dec_u32((u32)EMBED_DIM);
    uart1_puts(" mlp=");
    uart1_put_dec_u32((u32)MLP_DIM);
    uart1_puts("\r\n");
    boot_marker = 0x22222222U;
    Xil_DCacheEnable();
    boot_marker = 0x33333333U;
    stage("TINYVIT_UDP_PL_UART_BEGIN");
    preload_pl_static_params();
    scan_mdio_phy();

    if (net_init() != 0) {
        boot_marker = 0xE0000001U;
        uart1_puts("NET_INIT_FAIL\r\n");
        while (1);
    }
    boot_marker = 0x44444444U;

    uart1_puts("WAIT_UDP_IMAGE_784_BYTES\r\n");

    while (1) {
        boot_counter++;
        net_poll();
        poll_mdio_link_status();
        poll_gem_stats();

        if (udp_image_ready) {
            boot_marker = 0x55555555U;
            udp_image_ready = 0;
            run_uploaded_image();
            boot_marker = 0x44444444U;
            uart1_puts("WAIT_UDP_IMAGE_784_BYTES\r\n");
        }
    }
}







