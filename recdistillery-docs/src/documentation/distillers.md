# Distillers

Distillers define how teacher knowledge is transferred to the student model.
RecDistillery currently includes embedding-based, ranking-based, and composite
distillation strategies.

## Available Distillers

```text
DE
RRD
DE_RRD
HTD
FTD
UnKD
```

Distiller defaults are stored in:

```text
config/distillation/
```

## Distiller Base

::: recdistill.distillers.base

## DE

::: recdistill.distillers.de

## RRD

::: recdistill.distillers.rrd

## UnKD

::: recdistill.distillers.unkd

## HTD

::: recdistill.distillers.htd

## FTD

::: recdistill.distillers.ftd

## Composite Distillation

::: recdistill.distillers.composite

## Distillation Samplers

::: recdistill.samplers.base

::: recdistill.samplers.negative

::: recdistill.samplers.rrd

::: recdistill.samplers.teacher_topk
