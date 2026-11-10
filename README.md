# Meta-RL no Crafter com RL² + PPO

> **English summary.** Meta-RL on Crafter with RL² (GRU policy, PPO). Each trial samples 1 of 4
> hidden goals; the agent plays 3 episodes of 192 steps in fresh worlds and can only infer the goal
> from reward. Observation is a 549-number feature vector from the local 9×7 semantic map (not
> pixels); 6 of Crafter's 17 actions. Code, CLI and docs are in Portuguese. Result: the agent learns
> the task (~80% success) with a **fixed routine**, and recurrent memory showed **no measurable
> effect** under any of the 14 conditions tested.

**Autores:** <!-- PREENCHER: nomes dos autores -->
**Disciplina:** <!-- PREENCHER: nome da disciplina / professor / instituição / semestre -->
**Licença:** <!-- PREENCHER: ex. MIT, Apache-2.0, CC-BY-4.0 -->

---

## 1. O projeto

Trabalho final de uma disciplina de Deep RL do mestrado. A pergunta é se uma política recorrente
treinada por meta-RL usa de fato a memória para **identificar um objetivo oculto** a partir da
recompensa, ou se ela apenas aprende uma rotina fixa que funciona para todos os objetivos.

A abordagem é RL² ([Duan et al., 2016](https://arxiv.org/abs/1611.02779)): uma GRU recebe a
observação mais a ação, a recompensa e o `done` do passo anterior, e sua memória **atravessa** os
episódios de um mesmo trial. O PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))
atualiza os pesos entre lotes de trials; a adaptação *dentro* de um trial acontece só na memória,
sem atualização de pesos. Não há professor, imitação, rótulo do objetivo nem replay.

### A tarefa

| | |
|---|---|
| **Objetivos ocultos** | `collect_wood` (madeira), `collect_drink` (água), `eat_cow` (vaca), `collect_sapling` (muda) |
| **Trial** | 1 objetivo sorteado + K = 3 episódios de 192 passos, cada um num mundo novo |
| **Recompensa** | +1 por evento real do objetivo (opcionalmente com teto por episódio e penalidade por objetivo errado). O objetivo **nunca** é informado ao agente |
| **Observação** | vetor de 549 números (`45 + 63×8`) derivado **só** da janela 9×7 que o jogador vê: direção/distância do item mais próximo de cada tipo, o que está à frente, inventário, direção, luz. Não são pixels e não há mapa global |
| **Ações** | 6 das 17 do Crafter |
| **Rede** | GRU de 128 unidades, ator-crítico |

O agente só pode descobrir qual é o objetivo pela recompensa que recebe — e isso, em princípio,
deveria tornar o episódio 1 pior que os episódios 2 e 3.

### Como o efeito da memória é medido

Na avaliação, a mesma política é rodada em quatro modos sobre **os mesmos mundos e objetivos**:

- **`memory`** — a memória da GRU atravessa os 3 episódios (RL² de verdade);
- **`reset`** — a memória é zerada no início de cada episódio (mesmos pesos, sem informação entre episódios);
- **`untrained`** — pesos na inicialização;
- **`random`** — ações uniformes.

O **efeito da memória** é `memory − reset`, pareado episódio a episódio, com IC de 95%. Por
construção o episódio 1 é idêntico nos dois modos (a memória começa zerada de qualquer forma), então
só os episódios 2+ podem mostrar efeito. Essa comparação pareada é a medida central do trabalho:
ela isola a contribuição da memória de tudo o mais.

---

## 2. Estrutura de pastas

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/                        os 6 scripts (devem ficar juntos: eles se acham por __file__)
│   ├── crafter_meta_goals_rl2_ppo_v3.py   treino + avaliação (script principal)
│   ├── experimentos.py                    define e roda as fases A/B/C
│   ├── analisar.py                        tabelas, efeito da memória com IC, ranking
│   ├── diagnostico_comportamento.py       seletividade / matriz de comportamento
│   ├── gerar_videos.py                    vídeos comparando memory/reset/random
│   └── progresso.py                       acompanha uma fase em andamento
├── analises/                   curvas de treino de cada fase
│   ├── analise_A/curvas_treino.png        fase A
│   ├── analise_B/curvas_treino.png        fase B
│   ├── analise_C/curvas_treino.png        fase C, γ = 0,995
│   └── analise_C_G999/curvas_treino.png   fase C, γ = 0,999
└── docs/
    └── exemplo_C-MELHOR_s2_comparacao.mp4   memory / reset / random lado a lado
```

**Este repositório contém código e figuras, não dados.** O `.gitignore` exclui todos os `*.csv`,
além de `resultados/`, `resultados_C_G999/`, `resultados_smoke/`, `.venv/`, `__pycache__/`,
checkpoints `*.pt`, vídeos `*.mp4` (exceto o exemplo em `docs/`) e logs.

Ao rodar os experimentos, o `experimentos.py` cria `resultados/<ID>/s<semente>/<data_hora>/` com
`config.json`, `manifest.json`, `status.json`, `training.json`, `episodes.csv`, `summary.json`,
`adaptacao.png`, `comportamento.csv`, `comportamento_resumo.json`, `best.pt`, `last.pt` e `log.txt`,
e o `analisar.py` agrega isso nos CSVs descritos na seção 4. Nada disso é versionado.

Consequência prática: **os números da seção 5 estão nas tabelas deste README** — essa é a forma em
que os resultados ficam registrados aqui. Um clone limpo roda tudo do zero, mas só reproduz os
valores depois de refazer as fases (cada execução da fase C custa ~3 h de CPU).

---

## 3. Instalação

```bash
uv venv --python 3.12
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
source .venv/bin/activate
```

O `crafter` e o `gym` precisam de `PYTHONUTF8=1` na build em alguns sistemas. Se a instalação
falhar, use a alternativa autocontida: o script principal tem um cabeçalho
[PEP 723](https://peps.python.org/pep-0723/) com as dependências e as variáveis de build, então

```bash
uv run --python 3.12 src/crafter_meta_goals_rl2_ppo_v3.py --help
```

resolve tudo sozinho, sem venv. Tudo foi rodado em **CPU**.

---

## 4. Como rodar

Chame os scripts a partir da **raiz do repositório** — os defaults de `--base` são `resultados`,
relativos ao diretório atual.

### Teste rápido (~2 atualizações, não mede aprendizado)

```bash
python src/experimentos.py rodar --fase A --ids A-R2 --sementes 0 --smoke
```

### As três fases

```bash
# Fase A: 6 tipos de recompensa (300 iterações, 16 ambientes, 16 repetições de teste)
python src/experimentos.py rodar --fase A --sementes 0,1,2 --paralelo 4

# Fase B: 8 variações sobre a melhor recompensa da fase A
python src/experimentos.py rodar --fase B --referencia A-R1 --sementes 0,1,2 --paralelo 4

# Fase C: confirmação com sementes novas (600 iterações, 25 repetições, com vídeo)
python src/experimentos.py rodar --fase C --referencia A-R1 --melhor B-G995 --sementes 2,3,4 --paralelo 4
python src/experimentos.py rodar --fase C --referencia A-R1 --melhor B-G999 --sementes 2,3,4 \
    --paralelo 4 --base resultados_C_G999
```

Rodar o mesmo comando de novo é seguro: execuções completas são puladas e as interrompidas são
retomadas do último `last.pt`. Use `listar` para ver o estado e `comandos` para só imprimir os
comandos. Enquanto uma fase roda, `python src/progresso.py --fase B --sementes 0,1,2 --paralelo 4`
estima o tempo restante.

> Custo real: cada execução da fase C levou **~3 h de CPU**; a fase C inteira (6 execuções × 2
> valores de γ) levou ~20 h com 4 em paralelo.

### Análises (depois de rodar as fases)

O `analisar.py` recebe **pastas** (uma ou mais), não ids: cada pasta pode ser uma execução ou
qualquer pasta acima delas. Como `resultados/` não é versionado, estes comandos só funcionam depois
de rodar as fases acima. Foram eles que geraram exatamente os CSVs em `analises/`:

```bash
# Fase A: os 6 grupos de recompensa
python src/analisar.py resultados/A-R? --saida analises/analise_A --referencia A-R0

# Fase B: as 8 variações + os 6 grupos da fase A, para comparar (sem os grupos C)
python src/analisar.py resultados/A-R? resultados/B-* --saida analises/analise_B --referencia A-R1

# Fase C (γ = 0,995) e a repetição com γ = 0,999
python src/analisar.py resultados/C-MELHOR resultados/C-SEM \
    --saida analises/analise_C --referencia C-SEM
python src/analisar.py resultados_C_G999 --saida analises/analise_C_G999 --referencia C-SEM
```

Passar `resultados/` inteiro também funciona, mas junta as três fases num só relatório (com a fase C
dentro das tabelas da fase B), que não é o agrupamento usado nos resultados da seção 5.

Cada pasta de saída recebe `validacao.csv`, `sucesso_por_modo.csv`, `sucesso_por_objetivo.csv`,
`efeito_memoria.csv`, `comparacao_com_referencia.csv`, `comportamento_por_grupo.csv` e
`curvas_treino.png`. Só o PNG é versionado; os CSVs ficam de fora pelo `.gitignore`.

### Vídeos (precisa dos `best.pt`, ou seja, de treinar antes)

```bash
python src/gerar_videos.py --fase C --semente 2 --fps 6
```

---

## 5. Resultados

As tabelas abaixo são o registro dos resultados (os CSVs que as originaram não são versionados; veja
a seção 2). Cada uma é reproduzível rodando a fase correspondente e o `analisar.py`. "Validação" é o
sucesso médio no último episódio nas 5 últimas validações (métrica usada para *escolher*
configurações, sem olhar o teste); 3 sementes por grupo.

### Fase A — qual recompensa cria pressão para identificar o objetivo?

Hipóteses: recompensas repetíveis levam a uma estratégia fixa ("beber sempre"), e limitar quantos
eventos contam por episódio deveria eliminá-la (H1); penalizar objetivos errados deveria aumentar a
seletividade (H2).

| ID | Recompensa | Validação | Sucesso no teste, ep. 3 (`memory`) |
|---|---|---|---|
| **A-R1** | teto de 3 eventos por episódio | **86,7% ± 6,2** | **73,4%** |
| A-R2 | só o 1º evento de cada episódio | 85,0% ± 10,8 | 65,1% |
| A-R5 | sem teto + penalidade 0,1 | 78,3% ± 4,7 | 54,2% |
| A-R0 | original: +1 por evento, sem teto | 63,3% ± 10,3 | 57,8% |
| A-R3 | 1º evento + penalidade 0,1 | 48,3% ± 10,3 | 42,7% |
| A-R4 | 1º evento + penalidade 0,5 | **0,0% ± 0,0** | 39,6% |

Referências no episódio 3: `random` 43,8%, `untrained` 48,4%.

- O teto de 3 eventos (A-R1) foi a melhor recompensa e seguiu para a fase B.
- **A penalidade por objetivo errado não produziu seletividade, produziu passividade.** Com 0,5
  (A-R4) o agente aprende a não fazer nada: validação 0% e teste (39,6%) *abaixo* do aleatório.
  Com 0,1 (A-R3) ele empata com o aleatório. H2 foi rejeitada na direção oposta à prevista.
- **O efeito da memória já é nulo aqui.** Em todos os 6 grupos o IC95 de `memory − reset` cruza zero
  nos episódios 2 e 3; os pontos estimados são em geral *negativos* (ex. A-R2 ep. 3: −0,037
  [−0,089; +0,016]).

### Fase B — γ, horizonte, episódios por trial, entropia e o controle sem memória

| ID | Variação | Hipótese | Validação |
|---|---|---|---|
| **B-SEM** | **controle: sem memória** | H6 | **96,7% ± 4,7** |
| B-E003 | entropia 0,003 | H7 | 93,3% ± 6,2 |
| B-G995 | γ = 0,995 | H3 | 93,3% ± 6,2 |
| B-G999 | γ = 0,999 | H3 | 88,3% ± 4,7 |
| A-R1 | *(referência)* | — | 86,7% ± 6,2 |
| B-E03 | entropia 0,03 | H7 | 80,0% ± 8,2 |
| B-K2 | 2 episódios por trial | H5 | 80,0% ± 4,1 |
| B-K5 | 5 episódios por trial | H5 | 76,7% ± 11,8 |
| B-H96 | episódios de 96 passos | H4 | 75,0% ± 8,2 |

O resultado mais informativo é o do **controle**: o agente **sem memória** (`--no-memory`, mesma
arquitetura e mesmas entradas, GRU zerada a cada passo) ficou em **primeiro lugar**. Nenhuma das
manipulações pensadas para aumentar o valor da memória — γ maior (H3), episódios mais curtos (H4),
mais episódios por trial (H5), mais exploração (H7) — produziu efeito de memória com IC95 fora de
zero de forma consistente. As duas exceções pontuais em 32 comparações (B-G999 ep. 3:
+0,068 [+0,021; +0,120]; B-SEM ep. 2: +0,005 [0,000; +0,016]) não se sustentaram na fase C — e a
segunda é num grupo que **não tem memória**, o que indica ruído de amostragem, não sinal.

### Fase C — confirmação com sementes novas (2, 3, 4), 600 iterações, 25 repetições

`C-MELHOR` = A-R1 + o melhor γ, com memória. `C-SEM` = o mesmo, sem memória. 300 pares por
comparação.

| | | Validação | Teste ep. 3 (`memory`) | Efeito da memória, ep. 3 |
|---|---|---|---|---|
| γ = 0,995 | C-MELHOR | 93,3% ± 4,7 | 75,7% | −0,027 [−0,077; +0,023] |
| | C-SEM | **98,3% ± 2,4** | **83,7%** | — |
| γ = 0,999 | C-MELHOR | 95,0% ± 4,1 | **81,0%** | −0,027 [−0,073; +0,020] |
| | C-SEM | **98,3% ± 2,4** | 79,7% | — |

Referências: `random` ~44–53%, `untrained` ~43–45%.

**O efeito da memória é nulo nos dois valores de γ.** Todos os IC95 de `memory − reset` nos
episódios 2 e 3 contêm zero, e todos os pontos estimados são negativos ou praticamente zero
(γ = 0,999, ep. 2: −0,010 [−0,047; +0,027]). O agente com memória não superou o agente sem memória
em nenhuma das duas condições.

### Diagnóstico de comportamento — por que a memória não ajuda

A **seletividade** mede se o agente faz mais o evento do objetivo sorteado do que os outros (0 = faz
tudo igual, independentemente do objetivo). Vem do `comportamento_por_grupo.csv` que o `analisar.py`
gera para a fase C:

| | ep. 1 | ep. 2 | ep. 3 |
|---|---|---|---|
| C-MELHOR (γ 0,995), seletividade | −0,056 | −0,021 | +0,042 |
| C-MELHOR (γ 0,999), seletividade | −0,031 | −0,045 | −0,004 |

A seletividade fica **colada em zero e não cresce** entre os episódios. O indicador direto é
`bebidas_fora_agua`: o agente bebe água **7 a 9 vezes por episódio mesmo quando o objetivo não é
água**. Ou seja, ele executa a mesma rotina — beber, cortar madeira, procurar vaca — em todos os
trials. Essa rotina já resolve ~80% dos episódios, então **nunca há pressão de gradiente para
identificar o objetivo**, e sem essa pressão a memória não tem nada para carregar.

---

## 6. Conclusões

1. **O agente aprende a tarefa.** ~80% de sucesso contra ~44% do aleatório e ~44% da rede não
   treinada, de forma consistente em 3 sementes novas e dois valores de γ.

2. **Mas não aprende por meta-aprendizado.** Ele converge para uma **rotina fixa** que satisfaz
   qualquer um dos 4 objetivos, em vez de identificar qual é o objetivo e agir de acordo. A
   seletividade é ~0 e não cresce entre episódios.

3. **A memória recorrente não fez diferença em nenhuma condição testada** (14 configurações, 3
   fases, 2 valores de γ). Os IC95 de `memory − reset` contêm zero em todos os casos que se
   sustentaram, e o controle **sem memória** empatou ou ganhou — inclusive foi o 1º lugar da fase B
   e o melhor em validação na fase C.

4. **A causa é o desenho da tarefa, não a arquitetura.** Quatro objetivos atingíveis pela mesma
   rotina, num horizonte de 192 passos, não criam vantagem para identificar o objetivo. As tentativas
   de criar essa vantagem falharam de formas instrutivas: penalizar objetivos errados tornou o agente
   passivo (A-R4: validação 0%) em vez de seletivo, e limitar a recompensa repetível melhorou o
   desempenho sem produzir seletividade.

5. **Para um próximo passo**, a mudança teria que ser na tarefa: objetivos **mutuamente exclusivos**
   (em que buscar o errado impede o certo), ou custo de deslocamento que torne a rotina genérica
   inviável, ou objetivos inéditos no teste. Aumentar γ, o horizonte ou o número de episódios — as
   alavancas testadas aqui — não é suficiente.

### Limitações

- Observação semântica local e ciclo diurno modificado: **não** é o benchmark Crafter padrão (pixels).
- As 4 categorias de objetivo são compartilhadas entre treino, validação e teste (só as sementes dos
  mundos são separadas). Não é um teste de generalização para objetivos inéditos.
- 3 sementes por configuração; os desvios entre sementes são grandes o bastante para que diferenças
  de ~5 pontos de validação não sejam conclusivas.
- Treino em CPU, 300 iterações (fases A/B) e 600 (fase C). Não é um estudo de convergência.
