# Juiz do eval Se Liga (sem api-key) — prompt p/ Claude Code

O juiz é o Claude Code (subscrição, **sem ANTHROPIC_API_KEY**). Recebe o
`judge_package.json` produzido na box por `make_judge_package.py` (autossuficiente:
cada caso já traz o material de referência embutido).

Serve para dois modos com o MESMO prompt:
- **Interativo (agora):** `claude -p "$(cat judge.md)" --add-dir <pasta-com-judge_package.json>`
  (ou cola este conteúdo numa sessão do Claude Code apontada ao ficheiro).
- **Rotina cloud (depois):** mete este prompt numa routine `/schedule`; a box faz
  commit do `judge_package.json` no repo, a routine clona, lê e avalia.

---

## Instruções ao juiz

És um avaliador **RIGOROSO** de um chatbot de atendimento do "Se Liga na SKY"
(programa de incentivo — **não** é telecom/TV).

1. Lê `judge_package.json` (lista de casos). Cada caso tem: `id`, `query`,
   `answer` (resposta do bot), `reference` (material que deveria fundamentar a
   resposta), `expected_none`, `known_gap`.

2. Para cada caso, decide **pass** / **fail** com este critério:
   - A `answer` está **correta**, **fundamentada** no `reference` (não inventou
     factos fora dele) e **no escopo certo**?
   - Se `reference` começa por `(FORA DE ESCOPO…)`, só passa se a `answer` for a
     recusa padrão ("Não posso responder pois não está na minha base de dados.").
   - Se `reference` é `(skill não encontrada)` / `(doc não encontrado)`, marca
     fail com motivo "referência ausente".

3. Saída — replica o formato do harness, uma linha por caso:
   - `OK [id] query` — passou
   - `~ [id] query — motivo` — falhou MAS `known_gap: true` (não bloqueia)
   - `X [id] query — motivo` — falhou e `known_gap` é false (**bloqueia**)

4. No fim, imprime o sumário e o **veredito**:
   ```
   N/M passed (F failed, G lacuna(s) conhecida(s))
   VEREDITO: PASS        # se F == 0
   VEREDITO: FAIL        # se F > 0  -> não fazer cutover/deploy sem rever
   ```
   `F` conta só falhas com `known_gap: false`. O veredito FAIL é o "gate":
   equivale ao `sys.exit(1)` do harness — sinal para **não** promover o índice.

Não chames nenhuma API externa. Avalia só com o conteúdo do JSON.
