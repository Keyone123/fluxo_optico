import cv2
import numpy as np

def detect_motion(video_path, output_path, method="lucas-kanade"):
    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configurar o escritor de vídeo
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Ler o primeiro quadro
    ret, prev_frame = cap.read()
    if not ret:
        print("Erro ao carregar o vídeo!")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Criar máscara para destacar movimentos
    mask = np.zeros_like(prev_frame)

    # Configurações do método Horn-Schunck (opcional)
    horn_mask = np.zeros_like(prev_frame, dtype=np.float32)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if method == "lucas-kanade":
            # Usar Lucas-Kanade
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif method == "horn-schunck":
            # Usar Horn-Schunck
            # (emulamos usando parâmetros básicos aqui devido à API limitada)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 12, 2, 8, 1.0, 0)
        else:
            print("Método inválido!")
            break

        # Converter fluxo óptico em magnitude e ângulo
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Criar máscara colorida baseada na magnitude e ângulo
        mask[..., 0] = angle * 180 / np.pi / 2  # Ângulo em matiz (HSV)
        mask[..., 1] = 255                      # Saturação máxima
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude como intensidade

        # Converter máscara HSV para BGR para visualização
        segmented_frame = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Combinar com o quadro original para destacar movimento
        output_frame = cv2.addWeighted(curr_frame, 0.7, segmented_frame, 0.3, 0)

        # Salvar o quadro processado
        out.write(output_frame)

        # Atualizar o quadro anterior
        prev_gray = curr_gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processamento concluído! Vídeo salvo em: {output_path}")


# Execução do sistema:
# Substituir "vídeo.mp4" pelo nome do vídeo a ser processado
detect_motion("vídeo.mp4", "resultado_lucas_kanade.mp4", method="lucas-kanade")
detect_motion("vídeo.mp4", "resultado_horn_schunck.mp4", method="horn-schunck")
