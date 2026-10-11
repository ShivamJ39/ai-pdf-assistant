import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export interface UploadResponse {
  status: string;
  message: string;
}

export interface ChatResponse {
  answer: string | null;
}

@Injectable({ providedIn: 'root' })
export class PdfApiService {
  private readonly http = inject(HttpClient);

  private readonly baseUrl = 'https://ai-pdf-assistant-yoia.onrender.com';

  uploadPdf(file: File) {
    const formData = new FormData();

    // Your backend checks for a lowercase ".pdf" extension.
    const filename = file.name.replace(/\.pdf$/i, '.pdf');

    formData.append('file', file, filename);

    return this.http.post<UploadResponse>(
      `${this.baseUrl}/upload`,
      formData,
    );
  }

  askQuestion(question: string) {
    const formData = new FormData();
    formData.append('question', question);

    return this.http.post<ChatResponse>(
      `${this.baseUrl}/chat`,
      formData,
    );
  }
}