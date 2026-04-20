---
name: pag-contas-processando
description: Use quando o participante relatar que pagou uma conta (luz, água, etc.) pelo portal Se Liga na SKY e o status continua aparecendo como "processando", ou quando a conta paga não aparece como concluída no aplicativo.
---
# Skill: Pagamento de Contas em Processando

## Quando usar
- Participante diz que pagou uma conta pelo aplicativo/portal e aparece "processando".
- Participante diz que a conta não consta como paga mesmo após o resgate.
- Participante está preocupado que a conta não será paga no prazo.

## Regra de negócio

> As contas resgatadas via Se Liga na SKY são processadas e pagas **até o fim do dia** do vencimento. O status "processando" durante o dia é normal e não indica falha.

## Procedimento

1. **Confirmar o ICARE** do participante.
2. **Verificar no sistema** se o resgate foi realizado corretamente e se a data de vencimento é do dia.
3. **Tranquilizar o participante**: o status "processando" é o comportamento esperado durante o dia.
4. **Orientar** a verificar novamente ao fim do dia ou no dia seguinte.
5. **Se o dia já passou e ainda aparece processando**: escalar para verificação de falha no processamento.

## Mensagem padrão

> "As contas são processadas e pagas até o fim do dia. O status 'processando' é normal durante o dia — não significa que houve problema. Verifique novamente mais tarde."

## Como acessar o comprovante (após processamento)
Quando o participante perguntar como baixar o comprovante de pagamento:
1. Acessar o portal Se Liga na SKY.
2. Ir até a seção de **Resgates** ou **Histórico**.
3. Localizar a transação e clicar em **Comprovante** ou **Download**.

## Observações
- Contas com vencimento no dia corrente: processadas até às 23h59 do mesmo dia.
- Contas pagas via boleto têm prazo diferente — ver skill `antecipar-boleto`.
- Se o processamento não foi concluído até o dia seguinte útil, abrir chamado de investigação.

## Exemplo real (chat Lisa, 25/02/2025 e 26/02/2025)
- Participante: pagou conta de luz, aparecia "processando".
- Resolução: atendente confirmou que é processado até fim do dia e orientou a aguardar.
- Segundo chamado: conta do dia aparecia processando às 18h → orientação igual.
