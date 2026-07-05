"use client";

import { useRouter } from "next/navigation";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { enrollmentUrl } from "@/lib/kiosk";

const registrationSchema = z.object({
  name: z.string().trim().min(2, "Name is required."),
  empCode: z.string().trim().min(2, "Employee code is required."),
});

type RegistrationValues = z.infer<typeof registrationSchema>;

export function EmployeeRegistration() {
  const router = useRouter();
  const form = useForm<RegistrationValues>({
    resolver: zodResolver(registrationSchema),
    defaultValues: { name: "", empCode: "" },
  });

  const onSubmit = (values: RegistrationValues) => {
    router.push(enrollmentUrl(values.name, values.empCode));
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Register New Employee</CardTitle>
      </CardHeader>
      <CardContent>
        <form
          className="max-w-md space-y-5"
          onSubmit={form.handleSubmit(onSubmit)}
          noValidate
        >
          <div className="space-y-2">
            <Label htmlFor="employee-name">Name</Label>
            <Input
              id="employee-name"
              placeholder="Full name"
              aria-invalid={Boolean(form.formState.errors.name)}
              aria-describedby={
                form.formState.errors.name ? "employee-name-error" : undefined
              }
              {...form.register("name")}
            />
            {form.formState.errors.name && (
              <p
                id="employee-name-error"
                className="text-xs text-destructive"
                role="alert"
              >
                {form.formState.errors.name.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="employee-code">Employee Code</Label>
            <Input
              id="employee-code"
              placeholder="EMP-1004"
              aria-invalid={Boolean(form.formState.errors.empCode)}
              aria-describedby={
                form.formState.errors.empCode
                  ? "employee-code-error"
                  : undefined
              }
              {...form.register("empCode")}
            />
            {form.formState.errors.empCode && (
              <p
                id="employee-code-error"
                className="text-xs text-destructive"
                role="alert"
              >
                {form.formState.errors.empCode.message}
              </p>
            )}
          </div>

          <p className="text-xs text-muted-foreground">
            Nhập thông tin rồi tiếp tục sang màn hình kiosk để chụp khuôn mặt đăng
            ký.
          </p>

          <Button type="submit">Đăng ký trên Kiosk</Button>
        </form>
      </CardContent>
    </Card>
  );
}
