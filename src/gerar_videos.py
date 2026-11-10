"""Gera vídeos comparativos dos modelos já treinados (nada é treinado aqui).

Cada vídeo mostra, para cada objetivo escolhido, uma grade:
    linhas  = modos: memory (agente completo), reset (mesmo agente, memória apagada
              entre episódios) e random (ações ao acaso)
    colunas = episódios do trial (mesmo objetivo, mundos diferentes)
Em cada painel: passo, recompensa acumulada e uma moldura verde a partir do passo
em que o objetivo foi cumprido. Os três modos jogam exatamente os mesmos mundos.

Para cada experimento (ex.: A-R1) e semente, usa o best.pt da execução completa e
a configuração do próprio treino (recompensa, episódios, memória, rede, horizonte).
Os mundos são os mesmos das primeiras tarefas do teste (uma por objetivo).

Uso:
    python gerar_videos.py                          # todos os A-* da semente 0, 6 quadros/s
    python gerar_videos.py --ids A-R1,A-R2 --fps 3  # mais devagar
    python gerar_videos.py --objetivos madeira,vaca --semente 1
    python gerar_videos.py --fase B
Saída: PASTA_SAIDA/<id>_s<semente>_comparacao.mp4 (use --refazer para gerar de novo).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
NOMES = {'madeira': 'collect_wood', 'agua': 'collect_drink', 'vaca': 'eat_cow', 'muda': 'collect_sapling'}
TITULOS = {'madeira': 'MADEIRA', 'agua': 'ÁGUA', 'vaca': 'VACA', 'muda': 'MUDA'}
MODOS = (('memory', 'memory: agente completo', {}),
         ('reset', 'reset: memória apagada', {'reset_memory': True}),
         ('random', 'random: ações ao acaso', {'random_policy': True}))


def carregar_modulo(caminho):
    spec = importlib.util.spec_from_file_location('rl2_script', caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def carregar_modelo(m, checkpoint):
    estado = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if estado.get('version') != m.VERSION:
        raise SystemExit(f"{checkpoint}: versão {estado.get('version')}, script {m.VERSION}.")
    contrato = estado['contract']
    # Reproduz a recompensa e o número de episódios do treino (a recompensa anterior entra na GRU).
    m.TETO_RECOMPENSA = contrato.get('reward_cap', 0)
    m.PENALIDADE_ERRADO = contrato.get('wrong_penalty', 0.0)
    m.K = contrato.get('episodes_per_trial', 3)
    modelo = m.Agente(contrato['hidden'], sem_memoria=contrato.get('no_memory', False))
    modelo.load_state_dict(estado['model'])
    modelo.eval()
    return modelo, contrato


def execucao_completa(base, ident, semente):
    for status in sorted((base / ident / f's{semente}').glob('*/status.json'), reverse=True):
        try:
            if json.loads(status.read_text(encoding='utf-8')).get('training_complete'):
                return status.parent
        except (OSError, ValueError):
            continue
    return None


def fonte(tamanho):
    """DejaVu Sans (vem com o matplotlib) tem acentos; a fonte padrão do Pillow não."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), tamanho)
    except Exception:
        try:
            return ImageFont.load_default(size=tamanho)
        except TypeError:      # Pillow antigo: fonte fixa
            return ImageFont.load_default()


def gravar_video(m, modelo, contrato, ident, semente, objetivos, destino, fps, painel):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    k, h = m.K, contrato['horizon']
    margem, topo, rotulo = 225, 44, 22
    largura = margem + k * painel
    altura = topo + len(MODOS) * (painel + rotulo)
    largura += largura % 2
    altura += altura % 2
    f_titulo, f_texto = fonte(18), fonte(14)
    vetor = m.Vetor(1, h)
    try:
        with imageio.get_writer(destino, fps=fps, codec='libx264', macro_block_size=2) as video:
            for alvo in objetivos:
                g = m.OBJETIVOS.index(alvo)
                semente_mundo = 1_600_000_000 + g * 100003      # mesma tarefa g do conjunto de teste
                gravacoes = []
                for _, _, opcoes in MODOS:
                    torch.manual_seed(54321 + g)                 # mesmo sorteio para os três modos
                    dados, _, quadros = m.collect(vetor, modelo, [g], semente_mundo, record=True, **opcoes)
                    recompensas = dados['rewards'][:, 0].numpy().reshape(k, h)
                    gravacoes.append((quadros, recompensas))
                nome = [n for n, v in NOMES.items() if v == alvo][0]
                for t in range(h):
                    tela = Image.new('RGB', (largura, altura), '#12141c')
                    d = ImageDraw.Draw(tela)
                    d.text((12, 10), f'{ident} (semente {semente})   objetivo sorteado: {TITULOS[nome]} ({alvo})'
                                     f'   passo {t + 1}/{h}', fill='white', font=f_titulo)
                    for linha, ((_, legenda, _), (quadros, recompensas)) in enumerate(zip(MODOS, gravacoes)):
                        y0 = topo + linha * (painel + rotulo)
                        d.text((12, y0 + rotulo + painel // 2 - 10), legenda, fill='#d0d4e0', font=f_texto)
                        for ep in range(k):
                            x0 = margem + ep * painel
                            acumulada = float(recompensas[ep, :t + 1].sum())
                            positivos = np.flatnonzero(recompensas[ep] > 0)
                            acertou = len(positivos) and positivos[0] <= t
                            texto = f'ep. {ep + 1}   recompensa {acumulada:+.1f}'
                            if acertou:
                                texto += f'   objetivo no passo {positivos[0] + 1}'
                            d.text((x0 + 4, y0 + 3), texto, fill='#7ee08a' if acertou else 'white', font=f_texto)
                            imagem = Image.fromarray(quadros[ep][t]).resize((painel - 6, painel - 6))
                            tela.paste(imagem, (x0 + 3, y0 + rotulo + 3))
                            if acertou:
                                d.rectangle([x0 + 1, y0 + rotulo + 1, x0 + painel - 2, y0 + rotulo + painel - 2],
                                            outline='#3ccf5a', width=3)
                    quadro = np.asarray(tela)
                    repeticoes = int(fps) if t == 0 else 1       # pausa de ~1 s no início de cada objetivo
                    for _ in range(max(1, repeticoes)):
                        video.append_data(quadro)
    finally:
        vetor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fase', default='A', help='usa todos os experimentos que começam com esta letra')
    parser.add_argument('--ids', default='', help='ids separados por vírgula (substitui --fase)')
    parser.add_argument('--semente', type=int, default=0)
    parser.add_argument('--objetivos', default='madeira,agua,vaca,muda', help='quais objetivos mostrar, em ordem')
    parser.add_argument('--fps', type=float, default=6, help='quadros por segundo (o jogo dá 1 passo por quadro)')
    parser.add_argument('--painel', type=int, default=240, help='tamanho de cada painel, em pixels')
    parser.add_argument('--base', default='resultados')
    parser.add_argument('--saida', default='videos')
    parser.add_argument('--refazer', action='store_true', help='gera de novo mesmo se o vídeo já existir')
    parser.add_argument('--script', default=str(AQUI / 'crafter_meta_goals_rl2_ppo_v3.py'))
    args = parser.parse_args()

    desconhecidos = [o for o in args.objetivos.split(',') if o not in NOMES]
    if desconhecidos:
        raise SystemExit(f'Objetivos desconhecidos: {desconhecidos}. Use: {", ".join(NOMES)}')
    objetivos = [NOMES[o] for o in args.objetivos.split(',')]
    base, saida = Path(args.base).resolve(), Path(args.saida).resolve()
    ids = args.ids.split(',') if args.ids else sorted(p.name for p in base.glob(f'{args.fase}-*') if p.is_dir())
    if not ids:
        raise SystemExit(f'Nenhum experimento encontrado em {base}.')
    saida.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    m = carregar_modulo(args.script)

    gerados = []
    for ident in ids:
        execucao = execucao_completa(base, ident, args.semente)
        if execucao is None:
            print(f'[pulado]  {ident} s{args.semente}: nenhuma execução completa', flush=True)
            continue
        destino = saida / f'{ident}_s{args.semente}_comparacao.mp4'
        if destino.exists() and not args.refazer:
            print(f'[pronto]  {ident} s{args.semente}: {destino} (já existia)', flush=True)
            gerados.append(destino)
            continue
        print(f'[gerando] {ident} s{args.semente} ...', flush=True)
        modelo, contrato = carregar_modelo(m, execucao / 'best.pt')
        gravar_video(m, modelo, contrato, ident, args.semente, objetivos, destino, args.fps, args.painel)
        gerados.append(destino)
        print(f'[pronto]  {ident} s{args.semente}: {destino}', flush=True)
    print(f'\n{len(gerados)} vídeo(s) em {saida}')


if __name__ == '__main__':
    main()
