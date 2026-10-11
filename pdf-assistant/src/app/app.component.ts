import {
  afterNextRender,
  Component,
  computed,
  ElementRef,
  inject,
  Injector,
  signal,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { PdfApiService } from './services/pdf-api.service';

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string;
}

@Component({
  imports: [FormsModule],
  selector: 'app-root',
  standalone:true,
  styleUrl: './app.component.css',
  templateUrl: './app.component.html',
})
export class App {
  protected readonly title = signal('pdf-assistant');
  private readonly api = inject(PdfApiService);
  private readonly injector = inject(Injector);

  @ViewChild('chatLog')
  private chatLog?: ElementRef<HTMLDivElement>;

  readonly selectedFile = signal<File | null>(null);
  readonly activeDocument = signal('');
  readonly documentReady = signal(false);

  readonly uploading = signal(false);
  readonly asking = signal(false);
  readonly dragging = signal(false);

  readonly uploadMessage = signal('');
  readonly error = signal('');
  readonly question = signal('');
  readonly messages = signal<ChatMessage[]>([]);

  readonly busy = computed(() => this.uploading() || this.asking());

  readonly canSend = computed(
    () =>
      this.documentReady() &&
      !this.busy() &&
      this.question().trim().length > 0,
  );

  readonly suggestions = [
    'Summarize this document.',
    'What are the main takeaways?',
    'Explain the key concepts.',
  ];

  private nextMessageId = 0;

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];

    if (file) {
      this.selectFile(file);
    }

    // Allow selecting the same file again after an error.
    input.value = '';
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();

    if (!this.busy()) {
      this.dragging.set(true);
    }
  }

  onDragLeave(): void {
    this.dragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);

    if (this.busy()) {
      return;
    }

    const files = event.dataTransfer?.files;

    if (!files?.length) {
      return;
    }

    if (files.length > 1) {
      this.error.set('Please select one PDF at a time.');
      return;
    }

    this.selectFile(files[0]);
  }

  private selectFile(file: File): void {
    if (this.busy()) {
      return;
    }

    this.error.set('');

    if (!/\.pdf$/i.test(file.name)) {
      this.error.set('Please choose a PDF file.');
      return;
    }

    if (file.size === 0) {
      this.error.set('The selected file is empty.');
      return;
    }

    this.selectedFile.set(file);
    this.uploadMessage.set('');
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async uploadPdf(): Promise<void> {
    const file = this.selectedFile();

    if (!file || this.busy()) {
      return;
    }

    this.uploading.set(true);
    this.error.set('');
    this.uploadMessage.set('');

    // Require a successful upload before starting another conversation.
    this.documentReady.set(false);
    this.activeDocument.set('');
    this.messages.set([]);
    this.question.set('');

    try {
      const response = await firstValueFrom(this.api.uploadPdf(file));

      if (response.status !== 'success') {
        throw new Error(response.message || 'PDF processing failed.');
      }

      this.activeDocument.set(file.name);
      this.documentReady.set(true);
      this.selectedFile.set(null);
      this.uploadMessage.set(response.message);
    } catch (error: unknown) {
      this.error.set(this.getErrorMessage(error));
    } finally {
      this.uploading.set(false);
    }
  }

  useSuggestion(text: string): void {
    if (!this.documentReady() || this.busy()) {
      return;
    }

    this.question.set(text);
    void this.sendQuestion();
  }

  async sendQuestion(): Promise<void> {
    if (!this.canSend()) {
      return;
    }

    const question = this.question().trim();

    this.error.set('');
    this.asking.set(true);
    this.question.set('');

    this.addMessage('user', question);

    try {
      const response = await firstValueFrom(
        this.api.askQuestion(question),
      );

      if (!response.answer?.trim()) {
        throw new Error(
          'The AI returned an empty answer. Please try again.',
        );
      }

      this.addMessage('assistant', response.answer);
    } catch (error: unknown) {
      this.error.set(this.getErrorMessage(error));

      // Restore the question so the user can retry.
      this.question.set(question);

      if (error instanceof HttpErrorResponse && error.status === 400) {
        this.documentReady.set(false);
        this.activeDocument.set('');
      }
    } finally {
      this.asking.set(false);
      this.scrollToLatest();
    }
  }

  clearConversation(): void {
    if (this.busy()) {
      return;
    }

    this.messages.set([]);
    this.error.set('');
    this.question.set('');
  }

  private addMessage(
    role: ChatMessage['role'],
    text: string,
  ): void {
    this.messages.update((messages) => [
      ...messages,
      { id: ++this.nextMessageId, role, text },
    ]);

    this.scrollToLatest();
  }

  private scrollToLatest(): void {
    afterNextRender(
      () => {
        const element = this.chatLog?.nativeElement;

        if (element) {
          element.scrollTop = element.scrollHeight;
        }
      },
      { injector: this.injector },
    );
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return (
          'Cannot reach the backend. Check that FastAPI is running ' +
          'on port 8000 and that the API URL is correct.'
        );
      }

      const detail = error.error?.detail;

      if (typeof detail === 'string' && detail.trim()) {
        return detail;
      }

      if (Array.isArray(detail)) {
        return detail
          .map((item) => item.msg || 'Invalid request.')
          .join(' ');
      }

      if (error.status === 429) {
        return 'The AI rate limit was reached. Please wait and try again.';
      }

      if (error.status === 503) {
        return 'The AI service is busy. Please try again shortly.';
      }

      return `The request failed (${error.status}). Please try again.`;
    }

    if (error instanceof Error) {
      return error.message;
    }

    return 'Something went wrong. Please try again.';
  }
}
